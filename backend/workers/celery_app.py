from celery import Celery
from backend.config import get_settings

settings = get_settings()

celery_app = Celery(
    "ai_film_room",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["backend.workers.process_video_job"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # process one task at a time per worker
    task_soft_time_limit=3600,     # 1 hour soft limit
    task_time_limit=4200,          # 70 min hard limit
    result_expires=86400,          # 24h
)
