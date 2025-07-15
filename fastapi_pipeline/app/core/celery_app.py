# core/celery_app.py
from celery import Celery
from app.core.config import settings  

celery = Celery(
    "ray_triton_pipeline",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks"]
)

celery.conf.task_default_queue = "celery"


# celery = Celery(
#     "triton_pipeline",
#     broker=settings.REDIS_URL,
#     backend=settings.REDIS_URL  # Needed for task status tracking
# )

# celery.conf.update(
#     task_routes={
#         "app.tasks.*": {"queue": "default"},
#     },
#     task_serializer="json",
#     result_serializer="json",
#     accept_content=["json"],
# )
# # register or auto-discover  task module
# celery.autodiscover_tasks(["app.tasks"])
