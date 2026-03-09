from fastapi import FastAPI, UploadFile, File, Form, Body
from pymongo import MongoClient
from deepface import DeepFace
import numpy as np
import os
import json
import tempfile
import io
from collections import defaultdict
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

load_dotenv()

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# MongoDB Setup
# ==========================================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MONGO_URI not found in environment")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["studioDB"]
images_collection = db["images"]
customers_collection = db["customers"]
jobs_collection = db["jobs"]  # Track processing jobs

# ==========================================
# Load FaceNet Model at Startup
# ==========================================
@app.on_event("startup")
def startup():
    print("Loading Facenet512 model...")
    DeepFace.build_model("Facenet512")
    print("Model loaded ✅")

# ==========================================
# Health Check
# ==========================================
@app.get("/")
def home():
    return {"status": "DeepFace API running ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ==========================================
# Google Drive Service
# ==========================================
def get_drive_service():
    json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not json_str:
        raise Exception("GOOGLE_SERVICE_ACCOUNT_JSON not found")
    creds_dict = json.loads(json_str)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)

def extract_folder_id(folder_link: str) -> str:
    if "folders/" in folder_link:
        return folder_link.split("folders/")[1].split("?")[0].strip()
    return folder_link.strip()

# ==========================================
# LIST ALL FILES WITH PAGINATION — KEY FIX
# Pehle sirf 1st page aati thi, baaki miss hoti thi
# ==========================================
def list_all_drive_files(drive, folder_id: str) -> list:
    all_files = []
    page_token = None
    page_num = 0

    while True:
        page_num += 1
        params = {
            "q": f"'{folder_id}' in parents and mimeType contains 'image/' and trashed=false",
            "fields": "nextPageToken, files(id, name)",
            "pageSize": 1000,
        }
        if page_token:
            params["pageToken"] = page_token

        results = drive.files().list(**params).execute()
        files = results.get("files", [])
        all_files.extend(files)
        print(f"  Page {page_num}: {len(files)} files (total: {len(all_files)})")

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    return all_files

# ==========================================
# Cosine Similarity
# ==========================================
def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ==========================================
# Get Face Embeddings — Multi-strategy
# ==========================================
def get_embeddings(img_path: str) -> list:
    """
    Try 3 strategies so studio shots, selfies, angled faces all work.
    Returns list of face dicts sorted by face area (largest first).
    """
    for strategy in [
        {"enforce_detection": True,  "detector_backend": "retinaface"},
        {"enforce_detection": False, "detector_backend": "retinaface"},
        {"enforce_detection": False, "detector_backend": "opencv"},
    ]:
        try:
            results = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet512",
                **strategy
            )
            if results:
                results.sort(
                    key=lambda r: r.get("facial_area", {}).get("w", 0) *
                                  r.get("facial_area", {}).get("h", 0),
                    reverse=True
                )
                return results
        except Exception:
            continue

    raise Exception("No face detected in image")

# ==========================================
# Find natural score gap for threshold
# ==========================================
def find_score_gap(scores: list) -> float:
    if len(scores) < 2:
        return 0.55
    max_gap, pos = 0, 0
    for i in range(len(scores) - 1):
        gap = scores[i] - scores[i + 1]
        if gap > max_gap:
            max_gap, pos = gap, i
    if max_gap < 0.05:
        return 0.55
    return round((scores[pos] + scores[pos + 1]) / 2, 3)


# ==========================================
# PROCESS DRIVE FOLDER — Full Fixed Version
# Fixes: pagination, resume mode, batch insert, no crash on single fail
# ==========================================
@app.post("/process-drive-folder")
async def process_drive_folder(data: dict = Body(...)):
    try:
        folder_link = data.get("folder_link", "").strip()
        event_id    = data.get("event_id", "").strip()
        resume      = data.get("resume", False)

        if not folder_link or not event_id:
            return {"error": "folder_link and event_id are required"}

        drive     = get_drive_service()
        folder_id = extract_folder_id(folder_link)

        # --- Resume: skip already-done files ---
        already_done = set()
        if resume:
            docs = images_collection.find({"event_id": event_id}, {"file_id": 1})
            already_done = {d["file_id"] for d in docs}
            print(f"Resume mode: skipping {len(already_done)} already processed files")
        else:
            images_collection.delete_many({"event_id": event_id})
            print("Fresh run: cleared old embeddings")

        # --- Fetch ALL files (paginated) ---
        print(f"Listing Drive folder: {folder_id}")
        all_files    = list_all_drive_files(drive, folder_id)
        to_process   = [f for f in all_files if f["id"] not in already_done]

        print(f"Total in Drive: {len(all_files)} | To process: {len(to_process)} | Skipped: {len(already_done)}")

        processed   = 0
        failed      = 0
        total_faces = 0

        for i, file in enumerate(to_process):
            file_id   = file["id"]
            file_name = file["name"]
            print(f"[{i+1}/{len(to_process)}] {file_name}")

            try:
                # Download
                req    = drive.files().get_media(fileId=file_id)
                buf    = io.BytesIO()
                dl     = MediaIoBaseDownload(buf, req)
                done   = False
                while not done:
                    _, done = dl.next_chunk()
                buf.seek(0)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(buf.read())
                    tmp_path = tmp.name

                # Get embeddings
                try:
                    faces = get_embeddings(tmp_path)
                except Exception as fe:
                    print(f"  ⚠ No face: {fe}")
                    os.remove(tmp_path)
                    failed += 1
                    continue  # Don't stop — move to next file

                os.remove(tmp_path)

                image_url = f"https://drive.google.com/uc?export=view&id={file_id}"

                docs = []
                for idx, face in enumerate(faces):
                    docs.append({
                        "event_id":            event_id,
                        "file_name":           file_name,
                        "file_id":             file_id,
                        "image_url":           image_url,
                        "embedding":           face["embedding"],
                        "face_index":          idx,
                        "facial_area":         face.get("facial_area", {}),
                        "face_confidence":     face.get("face_confidence", 0),
                        "total_faces_in_image": len(faces),
                    })
                    total_faces += 1

                if docs:
                    images_collection.insert_many(docs)  # Batch insert — faster

                processed += 1
                print(f"  ✅ {len(faces)} face(s) saved")

            except Exception as e:
                print(f"  ❌ FAILED {file_name}: {e}")
                failed += 1
                continue  # Never stop on single failure

        return {
            "success":              True,
            "event_id":             event_id,
            "total_in_drive":       len(all_files),
            "skipped":              len(already_done),
            "attempted":            len(to_process),
            "processed":            processed,
            "failed":               failed,
            "total_faces_indexed":  total_faces,
        }

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        return {"error": str(e)}


# ==========================================
# MATCH FACE — Client selfie se photos dhundo
# ==========================================
@app.post("/match")
async def match_face(
    event_id:  str   = Form(...),
    name:      str   = Form(...),
    mobile:    str   = Form(...),
    file:      UploadFile = File(...),
    threshold: float = Form(default=0.55)
):
    try:
        # Save uploaded selfie
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            faces = get_embeddings(tmp_path)
            query_emb = faces[0]["embedding"]  # Largest face = the person
        except Exception as e:
            os.remove(tmp_path)
            return {"error": f"No face detected in selfie: {e}"}

        os.remove(tmp_path)

        # Load all stored embeddings for this event
        stored = list(images_collection.find({"event_id": event_id}))
        print(f"Scanning {len(stored)} embeddings for event: {event_id}")

        if not stored:
            return {"error": f"No embeddings found for event_id: {event_id}. Process the Drive folder first."}

        # Group by image file, find best matching face per image
        by_file = defaultdict(list)
        for doc in stored:
            by_file[doc["file_id"]].append(doc)

        best_per_image = []
        for file_id, faces in by_file.items():
            best_score = -1
            best_doc   = None
            for doc in faces:
                score = cosine_similarity(query_emb, doc["embedding"])
                if score > best_score:
                    best_score = score
                    best_doc   = doc
            best_per_image.append({
                "file_name":   best_doc["file_name"],
                "file_id":     file_id,
                "image_url":   best_doc["image_url"],
                "score":       round(best_score, 4),
                "face_index":  best_doc.get("face_index", 0),
                "faces_total": best_doc.get("total_faces_in_image", 1),
            })

        best_per_image.sort(key=lambda x: x["score"], reverse=True)
        matched    = [r for r in best_per_image if r["score"] >= threshold]
        top_scores = best_per_image[:15]

        suggested = find_score_gap([r["score"] for r in best_per_image])

        # Save customer record
        customers_collection.insert_one({
            "event_id":      event_id,
            "name":          name,
            "mobile":        mobile,
            "matches":       [r["image_url"] for r in matched],
            "total_matched": len(matched),
            "top_scores":    top_scores,
        })

        return {
            "success":               True,
            "matched_images":        [r["image_url"] for r in matched],
            "total_matched":         len(matched),
            "total_images_searched": len(by_file),
            "threshold_used":        threshold,
            "suggested_threshold":   suggested,
            "top_scores":            top_scores,
        }

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# DEBUG SCORES — Threshold tuning helper
# ==========================================
@app.post("/debug-scores")
async def debug_scores(
    event_id: str        = Form(...),
    file:     UploadFile = File(...)
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            faces     = get_embeddings(tmp_path)
            query_emb = faces[0]["embedding"]
        except Exception as e:
            os.remove(tmp_path)
            return {"error": f"No face detected: {e}"}

        os.remove(tmp_path)

        stored = list(images_collection.find({"event_id": event_id}))

        by_file = defaultdict(list)
        for doc in stored:
            by_file[doc["file_id"]].append(doc)

        best_per_image = []
        for file_id, faces in by_file.items():
            best_score, best_doc, all_scores = -1, None, []
            for doc in faces:
                score = cosine_similarity(query_emb, doc["embedding"])
                all_scores.append({"face_index": doc.get("face_index", 0), "score": round(score, 4)})
                if score > best_score:
                    best_score, best_doc = score, doc
            best_per_image.append({
                "file_name":      best_doc["file_name"],
                "image_url":      best_doc["image_url"],
                "best_score":     round(best_score, 4),
                "total_faces":    len(faces),
                "all_face_scores": all_scores,
            })

        best_per_image.sort(key=lambda x: x["best_score"], reverse=True)
        best_scores = [r["best_score"] for r in best_per_image]
        suggested   = find_score_gap(best_scores)

        return {
            "total_unique_images":   len(by_file),
            "total_embeddings":      len(stored),
            "suggested_threshold":   suggested,
            "best_per_image":        best_per_image,
            "distribution": {
                "above_0.80": sum(1 for s in best_scores if s > 0.80),
                "above_0.70": sum(1 for s in best_scores if s > 0.70),
                "above_0.60": sum(1 for s in best_scores if s > 0.60),
                "above_0.55": sum(1 for s in best_scores if s > 0.55),
                "above_0.50": sum(1 for s in best_scores if s > 0.50),
            },
        }

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# EVENT STATS
# ==========================================
@app.get("/event-stats/{event_id}")
def event_stats(event_id: str):
    total_emb    = images_collection.count_documents({"event_id": event_id})
    unique_files = len(images_collection.distinct("file_id", {"event_id": event_id}))
    pipeline = [
        {"$match": {"event_id": event_id}},
        {"$group": {"_id": "$file_id", "file_name": {"$first": "$file_name"}, "face_count": {"$sum": 1}}},
        {"$sort": {"face_count": -1}},
        {"$limit": 50},
    ]
    breakdown = list(images_collection.aggregate(pipeline))
    return {
        "event_id":            event_id,
        "unique_images":       unique_files,
        "total_embeddings":    total_emb,
        "multi_face_images":   sum(1 for b in breakdown if b["face_count"] > 1),
        "image_breakdown":     breakdown,
        "status":              "ready" if unique_files > 0 else "empty",
    }


# ==========================================
# PROCESSING STATUS
# ==========================================
@app.get("/processing-status/{event_id}")
def processing_status(event_id: str):
    unique = len(images_collection.distinct("file_id", {"event_id": event_id}))
    total  = images_collection.count_documents({"event_id": event_id})
    return {
        "event_id":       event_id,
        "images_indexed": unique,
        "face_embeddings": total,
        "status":         "ready" if unique > 0 else "empty",
    }


# ==========================================
# CLEANUP LOW CONFIDENCE
# ==========================================
@app.post("/cleanup-low-confidence")
async def cleanup_low_confidence(data: dict = Body(...)):
    try:
        event_id       = data.get("event_id")
        min_confidence = data.get("min_confidence", 0.9)
        if not event_id:
            return {"error": "event_id required"}
        result    = images_collection.delete_many({
            "event_id":        event_id,
            "face_confidence": {"$lt": min_confidence, "$exists": True}
        })
        remaining = images_collection.count_documents({"event_id": event_id})
        return {"removed": result.deleted_count, "remaining": remaining}
    except Exception as e:
        return {"error": str(e)}


# ==========================================
# TEST DB
# ==========================================
@app.get("/test-db")
def test_db():
    try:
        db.ping.insert_one({"ping": "ok"})
        return {"MongoDB": "connected ✅"}
    except Exception as e:
        return {"error": str(e)}
