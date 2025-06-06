from app.tasks.celery_config import app as celery_app

async def send_to_bot(message: str) -> str:
    task = celery_app.send_task("bot.process_message", args=[message])
    return task.get(timeout=10)