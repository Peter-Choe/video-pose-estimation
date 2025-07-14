from fastapi import FastAPI, UploadFile, File
from celery.result import AsyncResult
from app.core.config import settings
from .tasks import process_video,celery
import shutil
import uuid
import os

app = FastAPI()

@app.get("/")
def root():
    return {
        "debug": settings.DEBUG,
        "triton": settings.TRITON_URL,
        "redis": settings.REDIS_URL
    }


@app.post("/infer/")
async def upload_and_infer(file: UploadFile = File(...)):
    filename = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = process_video.delay(filename)
    return {"task_id": task.id}



@app.get("/celery-status/{task_id}")
async def get_task_status(task_id: str):
    result = AsyncResult(task_id, app=celery)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.successful() else None,
        "error": str(result.info) if result.failed() else None,
    }
