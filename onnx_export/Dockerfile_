FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Install required system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 wget git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install prebuilt mmcv + others
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html && \
    pip install -r requirements.txt 

# Optional: copy model/config files & ONNX exporter
COPY mmpose_models ./mmpose_models
COPY export_mmpose_to_onnx.py ./tools/deployment/

