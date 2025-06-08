from celery import Celery
import os
from bot_service.core.config import settings

def make_celery() -> Celery:
    """Фабрика для создания Celery приложения"""
    celery_app = Celery(
        "bot_service",
        broker=settings.REDIS_URL,
        backend=settings.REDIS_URL,
        include=['bot_service.tasks.bot']
    )
    

    celery_app.conf.update(

        task_track_started=True,
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        
        worker_concurrency=2,
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=50,
        
        # Таймауты
        task_time_limit=600,
        task_soft_time_limit=570,
        
        # Результаты
        result_expires=3600,
        task_ignore_result=False,
        
        task_routes={
            'bot_service.tasks.bot.process_question': {'queue': 'ml_queue'},
            'bot_service.tasks.bot.train_model': {'queue': 'training_queue'},
        },
        
        task_default_queue='ml_queue',
        task_queue_max_priority=10,
        task_default_priority=5,
    )
    
    return celery_app

celery_app = make_celery()

celery_app.autodiscover_tasks(['bot_service.tasks'])

if __name__ == '__main__':
    celery_app.start()