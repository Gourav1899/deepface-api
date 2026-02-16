from fastapi import FastAPI, UploadFile, File, Form
from pymongo import MongoClient
from deepface import DeepFace
import numpy as np
import os
from numpy import dot
from numpy.linalg import norm

app = FastAPI()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["studioDB"]

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

@app.get("/")
def home():
    return {"message": "API running"}

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

