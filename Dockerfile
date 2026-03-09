FROM python:3.10-slim

# System dependencies for OpenCV + DeepFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

# Pre-download FaceNet512 model at build time
# This means Render won't download it on every cold start
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512'); print('Model downloaded ✅')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
