# Base image with CUDA 11.8 and cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-setuptools python3-venv \
    libglib2.0-0 libgl1 git curl wget   unzip ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch (CUDA 11.8 compatible)
RUN pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

# Default command
CMD ["python"]
