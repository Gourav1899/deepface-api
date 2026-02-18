from fastapi import FastAPI, UploadFile, File, Form, Body
from pymongo import MongoClient
from deepface import DeepFace
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
import uuid
import json

from google.oauth2 import service_account
from googleapiclient.discovery import build

app = FastAPI()

# =============================
# MongoDB
# =============================

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

db = client["studioDB"]
images_collection = db["images"]
customers_collection = db["customers"]

# =============================
# Load model once
# =============================

@app.on_event("startup")
def startup():
    print("Loading FaceNet...")
    DeepFace.build_model("Facenet")
    print("Model ready")

# =============================
# Drive Service
# =============================

def get_drive_service():

    json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not json_str:
        raise Exception("GOOGLE_SERVICE_ACCOUNT_JSON missing")

    creds_dict = json.loads(json_str)

    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    return build("drive", "v3", credentials=creds)

# =============================
# Extract Folder ID
# =============================

def extract_folder_id(link):

    if "folders/" in link:
        return link.split("folders/")[1].split("?")[0]

    return link

# =============================
# Process Folder
# =============================

@app.post("/process-drive-folder")
async def process_drive_folder(data: dict = Body(...)):

    folder_link = data.get("folder_link")
    event_id = data.get("event_id")

    if not folder_link or not event_id:
        return {"error": "folder_link and event_id required"}

    try:

        folder_id = extract_folder_id(folder_link)

        drive = get_drive_service()

        files = drive.files().list(
            q=f"'{folder_id}' in parents and mimeType contains 'image/'",
            fields="files(id,name)",
            pageSize=1000
        ).execute().get("files", [])

        processed = 0
        failed = 0

        for file in files:

            file_id = file["id"]

            image_url = f"https://drive.google.com/uc?id={file_id}"

            try:

                response = requests.get(image_url, timeout=30)

                if response.status_code != 200:
                    failed += 1
                    continue

                image = Image.open(BytesIO(response.content)).convert("RGB")

                temp = f"/tmp/{uuid.uuid4()}.jpg"
                image.save(temp)

                embedding = DeepFace.represent(
                    img_path=temp,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]

                os.remove(temp)

                images_collection.insert_one({

                    "event_id": event_id,
                    "file_id": file_id,
                    "image_url": image_url,
                    "embedding": embedding

                })

                processed += 1

            except Exception as e:

                print("Image failed:", e)
                failed += 1

        return {

            "total": len(files),
            "processed": processed,
            "failed": failed

        }

    except Exception as e:

        return {"error": str(e)}

# =============================
# Match Face
# =============================

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/match")
async def match(
    event_id: str = Form(...),
    name: str = Form(...),
    mobile: str = Form(...),
    file: UploadFile = File(...)
):

    temp = f"/tmp/{uuid.uuid4()}.jpg"

    with open(temp, "wb") as f:
        f.write(await file.read())

    embedding = DeepFace.represent(
        img_path=temp,
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]

    os.remove(temp)

    matches = []

    for img in images_collection.find({"event_id": event_id}):

        score = cosine(
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

    return {"matches": matches}

# =============================
# Test
# =============================

@app.get("/")
def root():
    return {"status": "API running"}
