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

app = FastAPI()

# ===============================
# MongoDB Setup
# ===============================
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["studioDB"]

# ===============================
# Load Model at Startup (IMPORTANT)
# ===============================
@app.on_event("startup")
def load_model():
    print("Loading FaceNet model...")
    DeepFace.build_model("Facenet")
    print("Model Loaded Successfully")

# ===============================
# Cosine Similarity
# ===============================
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

@app.get("/")
def home():
    return {"message": "API running 🚀"}

# ==================================================
# Generate Embedding from Image URL
# ==================================================
@app.post("/generate-embedding")
async def generate_embedding(data: dict = Body(...)):
    try:
        image_url = data.get("image_url")
        event_id = data.get("event_id")

        if not image_url or not event_id:
            return {"error": "image_url and event_id required"}

        response = requests.get(image_url)

        if response.status_code != 200:
            return {"error": "Failed to download image"}

        # Save temp file safely
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(temp_filename)

        embeddings = DeepFace.represent(
            img_path=temp_filename,
            model_name="Facenet",
            enforce_detection=False
        )

        os.remove(temp_filename)

        if not embeddings:
            return {"faces_found": 0}

        embedding = embeddings[0]["embedding"]

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

# ==================================================
# Match Customer Face
# ==================================================
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

        embedding = DeepFace.represent(
            img_path=temp_filename,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        os.remove(temp_filename)

        images = db.images.find({"event_id": event_id})
        matched = []

        for img in images:
            score = cosine_similarity(
                np.array(embedding),
                np.array(img["embedding"])
            )

            if score > 0.65:   # Optimized threshold
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

@app.get("/test-db")
def test_db():
    try:
        db.events.insert_one({"test": "success"})
        return {"status": "MongoDB connected ✅"}
    except Exception as e:
        return {"error": str(e)}
