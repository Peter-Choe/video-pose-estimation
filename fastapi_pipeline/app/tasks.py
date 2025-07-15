from celery import Celery
import cv2
import numpy as np
from app.infer_client import infer_pose, infer_pose_batch  # import both
from app.core.config import settings
from app.core.logging_config import logger
from app.core.celery_app import celery

import os
import random
import time
import logging
from datetime import datetime

# Debug config print
print(f"[CONFIG DEBUG] ENV = {os.getenv('ENV')}, REDIS_URL = {settings.REDIS_URL}, USE_BATCH_INFER = {settings.USE_BATCH_INFER}")




# Batch size (can tune later)
BATCH_SIZE = 16

@celery.task
def process_video(video_path: str):
    print(f"[DEBUG] Task started with {video_path}")
    logger.info(f"Start Processing video '{video_path}' with settings: USE_BATCH_INFER = {settings.USE_BATCH_INFER} |  TRITON_URL: {settings.TRITON_URL}")

    if not os.path.exists(video_path):
        logger.warning(f"[FILE MISSING] '{video_path}' not found in Celery container.")
        return []

    cap = cv2.VideoCapture(video_path)
    results = []
    frame_batch = []
    start_time = time.time()

    orig_w, orig_h = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if orig_w is None:
            orig_h, orig_w = frame.shape[:2]

        if settings.USE_BATCH_INFER:
            # Accumulate frames for batching
            frame_batch.append(frame)

            if len(frame_batch) == BATCH_SIZE:
                keypoints_batch = infer_pose_batch(frame_batch, (orig_w, orig_h))
                results.extend(keypoints_batch)
                frame_batch = []
        else:
            # Single-frame inference mode
            keypoints = infer_pose(frame, orig_w, orig_h)
            results.append(keypoints)
            if random.random() < 0.1:
                print(f"{len(keypoints)} Keypoints detected: {keypoints}")

    # Process remaining frames if batching is on
    if settings.USE_BATCH_INFER and frame_batch:
        keypoints_batch = infer_pose_batch(frame_batch, (orig_w, orig_h))
        results.extend(keypoints_batch)
    
    from pprint import pformat

    cap.release()
    elapsed_time = time.time() - start_time
    # Log general info (like keypoints, task flow)
    logger.info(f"Processed video '{video_path}' in {elapsed_time:.2f} seconds")
  
    # Log only first 3 keypoints cleanly
    preview = results[:3]
    logger.info("Sample keypoints (first 3):\n%s", pformat(preview, width=100))
    total_keypoints = sum(len(kps) for kps in results)
    logger.info("Total frames processed: %d", len(results))
    logger.info("Total keypoints detected: %d", total_keypoints)
    # Log metrics for inference time 
    record = logger.makeRecord(
    name=logger.name,
    level=logging.INFO,
    fn="", lno=0, msg=(
        f"{datetime.now().isoformat()} | Time: {elapsed_time:.2f} seconds | "
        f"USE_BATCH_INFER = {settings.USE_BATCH_INFER} | BATCH_SIZE = {BATCH_SIZE} | "
        f"ENV: {settings.ENV} | Video: {video_path}"
        ),
        args=None,
        exc_info=None
    )
    record.metrics_only = True  # flag for metrics filter
    logger.handle(record)

    return results


#  DEBUG: Check the actual Celery instance ID in this module
print(f"[CELERY WORKER DEBUG] Celery instance ID in tasks.py: {id(celery)}")