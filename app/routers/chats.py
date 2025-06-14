from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from celery.result import AsyncResult
from bot_service.tasks.celery_app import celery_app

router = APIRouter(prefix="/bot", tags=["bot"])

class MessageRequest(BaseModel):
    question: str
    user_id: str | None = None

@router.post("/ask")
async def ask_bot(data: MessageRequest):
    task = celery_app.send_task(
        'bot_service.tasks.bot.process_question',
        args=[data.question, data.user_id],
        kwargs={}
    )
    return {"task_id": task.id}

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if res.state == "PENDING":
        return {"status": res.state}
    if res.state in ("PROCESSING", "STARTED"):
        return {"status": res.state, "meta": res.info}
    if res.state == "SUCCESS":
        return {"status": res.state, "result": res.result}
    if res.state == "FAILURE":
        raise HTTPException(status_code=500, detail=str(res.result))
    return {"status": res.state}
