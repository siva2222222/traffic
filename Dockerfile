FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1

# System deps:
#   ffmpeg        -> siren detection (audio.py extracts the audio track)
#   libgl1/libglib -> required by opencv-python even with no display attached
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU-only torch wheel (~190MB vs ~554MB for the CUDA build) from the dedicated PyTorch
# index, installed first and separately with retries since it's the biggest download and
# a slow/flaky connection otherwise times out mid-file and fails the whole build.
RUN pip install --retries 10 torch --index-url https://download.pytorch.org/whl/cpu

# torch is unpinned in requirements.txt, so this sees it already satisfied and won't
# re-pull the much larger default (CUDA) build.
RUN pip install --retries 10 -r requirements.txt

COPY . .

# Writable at runtime: uploaded videos/photos, saved evidence, sqlite db, and the
# YOLO/CLIP/EasyOCR model caches so they aren't re-downloaded on every container start.
RUN mkdir -p static/uploads static/evidence data /root/.cache
VOLUME ["/app/static/uploads", "/app/static/evidence", "/app/data", "/root/.cache"]

EXPOSE 8000
ENV PORT=8000

CMD ["python", "app.py"]
