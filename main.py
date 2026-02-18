from fastapi import FastAPI, UploadFile, File, Form, Body
from pymongo import MongoClient
from deepface import DeepFace
import numpy as np
import os
from io import BytesIO
from PIL import Image
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

images_collection = db["images"]
customers_collection = db["customers"]


# ==========================================
# Load Model at Startup
# ==========================================

@app.on_event("startup")
def startup():

    print("Loading FaceNet model...")
    DeepFace.build_model("Facenet")
    print("FaceNet model loaded successfully")


# ==========================================
# Health Check
# ==========================================

@app.get("/")
def home():

    return {"status": "DeepFace API running"}


@app.get("/health")
def health():

    return {"status": "ok"}


# ==========================================
# Google Drive Service Account Setup
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

    service = build("drive", "v3", credentials=creds)

    return service


# ==========================================
# Extract Folder ID from Link
# ==========================================

def extract_folder_id(folder_link):

    if "folders/" in folder_link:

        return folder_link.split("folders/")[1].split("?")[0]

    return folder_link


# ==========================================
# PROCESS ENTIRE GOOGLE DRIVE FOLDER
# ==========================================

@app.post("/process-drive-folder")
async def process_drive_folder(data: dict = Body(...)):

    try:

        folder_link = data.get("folder_link")
        event_id = data.get("event_id")

        if not folder_link or not event_id:

            return {"error": "folder_link and event_id required"}

        drive = get_drive_service()

        folder_id = extract_folder_id(folder_link)

        results = drive.files().list(

            q=f"'{folder_id}' in parents and mimeType contains 'image/'",

            fields="files(id,name)",

            pageSize=1000

        ).execute()

        files = results.get("files", [])

        processed = 0
        failed = 0

        print(f"Total files found: {len(files)}")

        for file in files:

            file_id = file["id"]

            try:

                # DOWNLOAD IMAGE USING GOOGLE DRIVE API
                request = drive.files().get_media(fileId=file_id)

                file_bytes = request.execute()

                image = Image.open(BytesIO(file_bytes)).convert("RGB")

                # SAVE TEMP FILE
                temp_path = f"/tmp/{uuid.uuid4()}.jpg"

                image.save(temp_path)

                # GENERATE EMBEDDING
                embedding = DeepFace.represent(

                    img_path=temp_path,

                    model_name="Facenet",

                    enforce_detection=False

                )[0]["embedding"]

                os.remove(temp_path)

                image_url = f"https://drive.google.com/file/d/{file_id}/view"

                # SAVE TO MONGODB
                images_collection.insert_one({

                    "event_id": event_id,

                    "file_id": file_id,

                    "image_url": image_url,

                    "embedding": embedding

                })

                processed += 1

                print(f"Processed: {file_id}")

            except Exception as e:

                print("FAILED:", e)

                failed += 1


        return {

            "total_images": len(files),

            "processed": processed,

            "failed": failed

        }


    except Exception as e:

        print("ERROR:", e)

        return {"error": str(e)}


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==========================================
# MATCH FACE
# ==========================================

@app.post("/match")
async def match_face(

    event_id: str = Form(...),

    name: str = Form(...),

    mobile: str = Form(...),

    file: UploadFile = File(...)

):

    try:

        temp_path = f"/tmp/{uuid.uuid4()}.jpg"

        with open(temp_path, "wb") as f:

            f.write(await file.read())

        embedding = DeepFace.represent(

            img_path=temp_path,

            model_name="Facenet",

            enforce_detection=False

        )[0]["embedding"]

        os.remove(temp_path)

        matches = []

        images = images_collection.find({"event_id": event_id})

        for img in images:

            score = cosine_similarity(

                np.array(embedding),

                np.array(img["embedding"])

            )

            if score > 0.65:

                matches.append(img["image_url"])


        customers_collection.insert_one({

            "event_id": event_id,

            "name": name,

            "mobile": mobile,

            "matches": matches

        })

        return {

            "matched_images": matches

        }


    except Exception as e:

        return {"error": str(e)}


# ==========================================
# TEST MONGODB
# ==========================================

@app.get("/test-db")
def test_db():

    try:

        db.test.insert_one({"status": "ok"})

        return {"MongoDB": "connected"}

    except Exception as e:

        return {"error": str(e)}
