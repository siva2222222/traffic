FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install matching CPU-only PyTorch wheels. Keep these versions aligned with requirements.txt.
RUN pip install --retries 10 torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --retries 10 -r requirements.txt

# Fail the image build early if the torchvision NMS operator is unavailable.
RUN python -c "import torch, torchvision; from torchvision.ops import nms; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('NMS OK')"

COPY . .
RUN mkdir -p static/uploads static/evidence data /root/.cache

EXPOSE 8000
CMD ["python", "app.py"]
