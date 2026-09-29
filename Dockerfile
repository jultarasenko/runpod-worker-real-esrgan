ARG CUDA_VERSION="12.4.1"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      python3.10-venv \
      fonts-dejavu-core \
      rsync \
      git \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      libgoogle-perftools-dev \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Install the models
WORKDIR /workspace
RUN mkdir -p /workspace/models/ESRGAN && \
    cd /workspace/models/ESRGAN && \
    # Download the official Real-ESRGAN models
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
    # Compact per-frame network for video — 2.4MB against the photo models' 64MB
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth && \
    # Download additional models from Huggingface
    # wget https://huggingface.co/snappic/upscalers/resolve/main/4x-UltraSharp.pth && \
    # wget https://huggingface.co/snappic/upscalers/resolve/main/lollypop.pth && \
    # Download the GFPGAN models
    mkdir -p /workspace/models/GFPGAN && \
    wget -O /workspace/models/GFPGAN/GFPGANv1.3.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

# Install Torch
ARG INDEX_URL="https://download.pytorch.org/whl/cu124"
ARG TORCH_VERSION="2.6.0+cu124"
RUN pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL}

# 1. Clone the worker repo
# 2. Install requirements
# 3. Setup the local requirements
# 4. Create test_input.json file for test inference
# 5. Run test inference using handler.py to cache the models
ENV WORKER_BUILD_ENV="docker"
RUN git clone https://github.com/jultarasenko/runpod-worker-real-esrgan.git && \
    cd runpod-worker-real-esrgan && \
    pip3 install git+https://github.com/XPixelGroup/BasicSR.git && \
    pip3 install -r requirements.txt && \
    pip3 install -e . --no-deps && \
    python3 create_test_json.py && \
    python3 -u handler.py

# Overlay this fork's handler on the cloned checkout.
ADD handler.py /workspace/runpod-worker-real-esrgan/handler.py
ADD schemas/input.py /workspace/runpod-worker-real-esrgan/schemas/input.py

# Start the handler directly. The upstream start.sh symlinks a network volume over
# /workspace first, which this image does not use: the weights and the code are baked in.
WORKDIR /workspace/runpod-worker-real-esrgan
ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "handler.py"]
