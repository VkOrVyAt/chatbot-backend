from pydantic import BaseModel
from datetime import datetime

class ChatMessage(BaseModel):
    question: str
    answer: str

class ChatHistory(BaseModel):
    user_id: int

class ChatRead(BaseModel):
    id: int
    timestamp: datetime

    class Config:
        orm_mode = True