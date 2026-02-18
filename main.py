from fastapi import FastAPI, UploadFile, File, Form, Body
from pymongo import MongoClient
from deepface import DeepFace
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
from numpy import dot
from numpy.linalg import norm
import uuid
import json

from google.oauth2 import service_account
from googleapiclient.discovery import build

app = FastAPI()

# ==========================================
# MongoDB Setup
# ==========================================
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["studioDB"]

# ==========================================
# Load FaceNet Model at Startup
# ==========================================
@app.on_event("startup")
def load_model():
    print("Loading FaceNet model...")
    DeepFace.build_model("Facenet")
    print("Model Loaded Successfully")

# ==========================================
# Cosine Similarity
# ==========================================
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# ==========================================
# Extract Folder ID from Link
# ==========================================
def extract_folder_id(folder_link):
    if "folders/" in folder_link:
        return folder_link.split("folders/")[1].split("?")[0]
    return folder_link

# ==========================================
# Google Drive Service from ENV JSON
# ==========================================
def get_drive_service():
    service_account_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"))

    creds = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    service = build("drive", "v3", credentials=creds)
    return service

# ==========================================
# Home
# ==========================================
@app.get("/")
def home():
    return {"message": "DeepFace API Running 🚀"}

# ==========================================
# Process Entire Drive Folder (Bulk Embedding)
# ==========================================
@app.post("/process-drive-folder")
async def process_drive_folder(data: dict = Body(...)):
    try:
        folder_link = data.get("folder_link")
        event_id = data.get("event_id")

        if not folder_link or not event_id:
            return {"error": "folder_link and event_id required"}

        folder_id = extract_folder_id(folder_link)

        service = get_drive_service()

        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType contains 'image/'",
            fields="files(id, name)",
            pageSize=1000
        ).execute()

        files = results.get("files", [])

        processed = 0
        failed = 0

        for file in files:
            file_id = file["id"]
            image_url = f"https://drive.google.com/uc?export=download&id={file_id}"

            try:
                response = requests.get(image_url)

                if response.status_code != 200:
                    failed += 1
                    continue

                temp_filename = f"temp_{uuid.uuid4()}.jpg"

                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(temp_filename)

                embeddings = DeepFace.represent(
                    img_path=temp_filename,
                    model_name="Facenet",
                    enforce_detection=False
                )

                os.remove(temp_filename)

                if embeddings:
                    embedding = embeddings[0]["embedding"]

                    db.images.insert_one({
                        "event_id": event_id,
                        "image_url": image_url,
                        "embedding": embedding
                    })

                    processed += 1
                else:
                    failed += 1

            except Exception:
                failed += 1
                continue

        return {
            "total_images": len(files),
            "processed": processed,
            "failed": failed
        }

    except Exception as e:
        return {"error": str(e)}

# ==========================================
# Match Customer Face
# ==========================================
@app.post("/match")
async def match_face(
    event_id: str = Form(...),
    name: str = Form(...),
    mobile: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        temp_filename = f"temp_{uuid.uuid4()}.jpg"

        with open(temp_filename, "wb") as f:
            f.write(await file.read())

        embeddings = DeepFace.represent(
            img_path=temp_filename,
            model_name="Facenet",
            enforce_detection=False
        )

        os.remove(temp_filename)

        if not embeddings:
            return {"matched_images": []}

        embedding = embeddings[0]["embedding"]

        images = db.images.find({"event_id": event_id})

        matched = []

        for img in images:
            score = cosine_similarity(
                np.array(embedding),
                np.array(img["embedding"])
            )

            if score > 0.65:
                matched.append(img["image_url"])

        db.customers.insert_one({
            "event_id": event_id,
            "name": name,
            "mobile": mobile,
            "matched_images": matched
        })

        return {"matched_images": matched}

    except Exception as e:
        return {"error": str(e)}

# ==========================================
# Test DB
# ==========================================
@app.get("/test-db")
def test_db():
    try:
        db.events.insert_one({"test": "success"})
        return {"status": "MongoDB connected ✅"}
    except Exception as e:
        return {"error": str(e)}
