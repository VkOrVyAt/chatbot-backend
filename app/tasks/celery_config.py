from celery import Celery
from app.core.config import settings
import logging

# Настройка логирования для удобства отладки
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery(
    "chatbot",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    worker_concurrency=4,
    task_time_limit=300,
    task_soft_time_limit=280,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_track_started=True,
    result_expires=3600,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

logger.info("Celery успешно настроен с Redis как брокер и бэкенд")