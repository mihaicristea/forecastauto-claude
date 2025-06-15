FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create cache directories
RUN mkdir -p /app/models/cache/rembg

# Download models
RUN python3 models/download_models.py

# Pre-cache rembg models
RUN python3 scripts/cache_rembg.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TRANSFORMERS_CACHE=/app/models/cache
ENV HF_HOME=/app/models/cache
ENV HUGGINGFACE_HUB_CACHE=/app/models/cache
ENV U2NET_HOME=/app/models/cache/rembg
ENV PYTHONWARNINGS=ignore
ENV TF_CPP_MIN_LOG_LEVEL=2

CMD ["python3", "src/main.py"]

