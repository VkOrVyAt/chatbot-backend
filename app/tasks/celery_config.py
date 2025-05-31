from celery import Celery
from app.core.config import settings

app = Celery("chatbot", broker=settings.REDIS_URL, backend=settings.REDIS_URL)