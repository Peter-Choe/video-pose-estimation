from celery import Celery
import cv2
import numpy as np
from app.infer_client import infer_pose  
from app.core.config import settings

import os
import random
print(f"[CONFIG DEBUG] ENV = {os.getenv('ENV')}, REDIS_URL = {settings.REDIS_URL}")

import time
import logging
from datetime import datetime

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("logs/pose_inference.log"),
        logging.StreamHandler()
    ]
)


# For local development, you can use:
celery = Celery(__name__, broker=settings.REDIS_URL)


@celery.task
def process_video(video_path: str):
    logging.info(f"Start Processing video '{video_path}' with settings: {settings.ENV}, TRITON_URL: {settings.TRITON_URL}")

    cap = cv2.VideoCapture(video_path)
    results = []
    start_time = time.time()
    import random

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        orig_h, orig_w = frame.shape[:2]
        keypoints = infer_pose(frame, orig_w, orig_h)
        if  random.random() < 0.1:
            print(f"{len(keypoints)} Keypoints detected: {keypoints}")
        results.append(keypoints)
    cap.release()
    elapsed_time = time.time() - start_time
    logging.info(f"Processed video '{video_path}' in {elapsed_time:.2f} seconds")

    with open("logs/inference_metrics.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} | Time: {elapsed_time:.2f} seconds | ENV: {settings.ENV} | Video: {video_path} \n")

    return results
