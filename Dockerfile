FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 
    PYTHONDONTWRITEBYTECODE=1 
    PIP_DEFAULT_TIMEOUT=120 
    PIP_NO_CACHE_DIR=1

# System dependencies for OpenCV and audio extraction
RUN apt-get update && apt-get install -y --no-install-recommends 
    ffmpeg 
    libgl1 
    libglib2.0-0 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install a matching PyTorch and Torchvision CPU-only pair
RUN pip install --retries 10 --no-cache-dir 
    torch==2.5.1 torchvision==0.20.1 
    --index-url https://download.pytorch.org/whl/cpu

# Install the remaining application dependencies
RUN pip install --retries 10 --no-cache-dir -r requirements.txt

# Verify Torchvision NMS during the image build
RUN python -c "import torch, torchvision; from torchvision.ops import nms; print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__, 'NMS OK')"

COPY . .

# Runtime directories
RUN mkdir -p static/uploads static/evidence data /root/.cache

VOLUME ["/app/static/uploads", "/app/static/evidence", "/app/data", "/root/.cache"]

EXPOSE 8000

ENV PORT=8000

CMD ["python", "app.py"]