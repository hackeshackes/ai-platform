"""
Celery任务队列 v2.0
"""
from celery import Celery
from celery.schedules import crontab
from datetime import timedelta

celery_app = Celery(
    "ai_platform",
    broker="redis://localhost:6379/1",
    backend="redis://localhost:6379/2",
    include=["tasks.training", "tasks.inference", "tasks.data", "tasks.monitoring"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3000,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    max_retries=3,
    worker_prefetch_multiplier=4,
    worker_concurrency=4,
    task_routes={
        "tasks.training.*": {"queue": "training"},
        "tasks.inference.*": {"queue": "inference"},
        "tasks.data.*": {"queue": "data"},
        "tasks.notify.*": {"queue": "notification"},
        "tasks.monitoring.*": {"queue": "monitoring"},
    },
    beat_schedule={
        "sync-gpu-metrics": {
            "task": "tasks.monitoring.sync_gpu",
            "schedule": timedelta(seconds=5),
            "options": {"queue": "monitoring"},
        },
        "cleanup-tasks": {
            "task": "tasks.maintenance.cleanup",
            "schedule": crontab(hour=3, minute=0),
            "options": {"queue": "maintenance"},
        },
    },
)

class TaskPriority:
    CRITICAL = 0
    HIGH = 1
    NORMAL = 5
    LOW = 9

if __name__ == "__main__":
    celery_app.start()
