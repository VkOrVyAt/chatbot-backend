from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import List, Optional
import uuid

class ChatMessageCreate(BaseModel):
    question: str
    session_id: Optional[str] = None
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError('Вопрос слишком короткий')
        if len(v) > 1000:
            raise ValueError('Вопрос слишком длинный')
        return v.strip()
    
    @field_validator('session_id', mode='before')
    @classmethod
    def validate_session_id(cls, v):
        return v or str(uuid.uuid4())

class ChatMessageRead(BaseModel):
    id: int
    question: str
    answer: str
    created_at: datetime
    session_id: str
    
    model_config = {"from_attributes": True}

class ChatHistoryRead(BaseModel):
    messages: List[ChatMessageRead]
    total_count: int
    session_count: int

class ChatSessionRead(BaseModel):
    session_id: str
    message_count: int
    last_message_at: datetime
    first_message_at: datetime