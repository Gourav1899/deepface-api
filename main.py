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

app = FastAPI()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["studioDB"]

# cosine similarity function
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


@app.get("/")
def home():
    return {"message": "API running"}


# ============================================
# NEW ENDPOINT: generate embedding from image_url
# ============================================
@app.post("/generate-embedding")
async def generate_embedding(data: dict = Body(...)):

    try:
        image_url = data.get("image_url")
        access_token = data.get("access_token")
        event_id = data.get("event_id")

        if not image_url:
            return {"error": "image_url required"}

        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        # download image
        response = requests.get(image_url, headers=headers)

        if response.status_code != 200:
            return {"error": "Failed to download image"}

        # save temp image
        image = Image.open(BytesIO(response.content))
        image_np = np.array(image)

        # generate embedding
        embeddings = DeepFace.represent(
            img_path=image_np,
            model_name="Facenet",
            enforce_detection=False
        )

        if not embeddings:
            return {"faces_found": 0}

        embedding = embeddings[0]["embedding"]

        # save in MongoDB
        db.images.insert_one({
            "event_id": event_id,
            "image_url": image_url,
            "embedding": embedding
        })

        return {
            "faces_found": 1,
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================
# EXISTING ENDPOINT: match customer face
# ============================================
@app.post("/match")
async def match_face(
    event_id: str = Form(...),
    name: str = Form(...),
    mobile: str = Form(...),
    file: UploadFile = File(...)
):

    with open("temp.jpg", "wb") as f:
        f.write(await file.read())

    embedding = DeepFace.represent(
        img_path="temp.jpg",
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]

    images = db.images.find({"event_id": event_id})

    matched = []

    for img in images:
        score = cosine_similarity(
            np.array(embedding),
            np.array(img["embedding"])
        )
        if score > 0.75:
            matched.append(img["image_url"])

    db.customers.insert_one({
        "event_id": event_id,
        "name": name,
        "mobile": mobile,
        "matched_images": matched
    })

    return {"matched_images": matched}


@app.get("/test-db")
def test_db():
    try:
        db.events.insert_one({"test": "success"})
        return {"status": "MongoDB connected"}
    except Exception as e:
        return {"error": str(e)}
