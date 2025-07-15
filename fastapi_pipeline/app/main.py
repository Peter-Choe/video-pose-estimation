from fastapi import FastAPI, UploadFile, File
from celery.result import AsyncResult
from app.core.config import settings
from app.tasks import process_video
import uuid
import os
from pathlib import Path
from app.core.logging_config import logger

import os
print(f"[LOG DEBUG] CWD: {os.getcwd()}")
logger.info("Logger initialized and writing to video_pose_inference.log")
print(f"[CELERY DEBUG] Celery instance ID: {id(process_video.app)}")


app = FastAPI()

@app.get("/")
def root():
    return {
        "debug": settings.DEBUG,
        "triton": settings.TRITON_URL,
        "redis": settings.REDIS_URL
    }

async def save_upload_file(upload_file: UploadFile, destination: Path):
    with open(destination, "wb") as buffer:
        while chunk := await upload_file.read(1024 * 1024):  # 1MB
            buffer.write(chunk)
    logger.info(f"File saved to: {destination} ({destination.stat().st_size / 1024 / 1024:.2f} MB)")

@app.post("/infer/")
async def upload_and_infer(file: UploadFile = File(...)):
    filename = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
    video_path = Path(filename)
    await save_upload_file(file, video_path)

    logger.info(f"Dispatching video for inference: {video_path}")
    task = process_video.delay(str(video_path))

    return {"task_id": task.id}

#비동기 작업 상태 조회용
# @app.get("/celery-status/{task_id}")
# async def get_task_status(task_id: str):
#     result = AsyncResult(task_id, app=celery)
#     return {
#         "task_id": task_id,
#         "status": result.status,
#         "result": result.result if result.successful() else None,
#         "error": str(result.info) if result.failed() else None,
#     }
