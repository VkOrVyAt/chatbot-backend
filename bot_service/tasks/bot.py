from celery import Celery
from bot_service.core.config import settings

# Инициализация Celery
celery_app = Celery(
    "bot_service",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Настройки Celery
celery_app.conf.update(
    task_track_started=True,
    worker_concurrency=4,
    task_time_limit=300,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    task_serializer="json",
    accept_content=["json"],
)

@celery_app.task
def process_question(user_id: int, question: str):
    answer = f"Ответ на вопрос '{question}' для пользователя {user_id}"
    return answer