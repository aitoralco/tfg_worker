# Dockerfile for tfg_worker
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /worker

# Dependencias del sistema:
#   ffmpeg          → conversión de vídeo anotado a mp4
#   libgl1          → OpenCV (cv2)
#   libglib2.0-0    → OpenCV (cv2)
#   libsm6 libxext6 → OpenCV headless support
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de procesamiento de vídeo (fuera de /tmp para no usar RAM)
RUN mkdir -p /video_processing

COPY ./requirements.txt /worker/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /worker/requirements.txt

COPY . /worker

# Arrancar el worker RQ
CMD ["python", "-m", "app.main"]
