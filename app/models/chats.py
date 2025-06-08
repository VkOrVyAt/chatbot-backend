from sqlalchemy import Column, Integer, ForeignKey, DateTime, func, Index, Boolean, JSON, String
from sqlalchemy.orm import relationship
from sqlalchemy.ext.mutable import MutableList
from app.db.base import Base
from datetime import datetime
from typing import List, Dict, Any
import json

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    messages = Column(JSON, default=list, nullable=False)
    # Структура: [
    #   {"role": "user", "content": "Как работает доставка?", "timestamp": "2024-01-01T10:00:00"},
    #   {"role": "assistant", "content": "Доставка работает...", "timestamp": "2024-01-01T10:00:01"}
    # ]
    
    title = Column(String(200), nullable=True)
    status = Column(String(20), default="active", nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    user = relationship("User", back_populates="chat")

    __table_args__ = (
        Index('idx_chat_user', 'user_id'),
        Index('idx_chat_status_updated', 'status', 'updated_at'),
    )

    def add_message(self, role: str, content: str) -> None:
        """Добавляет сообщение в чат"""
        if self.messages is None:
            self.messages = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.messages = self.messages + [message]
        
        if not self.title and role == "user":
            self.title = content[:100] + "..." if len(content) > 100 else content

    def get_conversation_for_ai(self) -> List[Dict[str, str]]:
        """Возвращает всю беседу для AI (без timestamp)"""
        if not self.messages:
            return []
        
        return [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in self.messages
        ]

    def get_last_user_question(self) -> str:
        """Возвращает последний вопрос пользователя"""
        if not self.messages:
            return ""
        
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def mark_resolved(self) -> None:
        """Помечает чат как решенный"""
        self.status = "resolved"

    def restart_conversation(self) -> None:
        """Начинает новую беседу (очищает историю)"""
        self.messages = []
        self.title = None
        self.status = "active"

    def get_conversation_summary(self) -> Dict:
        """Возвращает краткую сводку чата"""
        if not self.messages:
            return {"questions": 0, "responses": 0, "last_activity": None}
        
        questions = sum(1 for msg in self.messages if msg["role"] == "user")
        responses = sum(1 for msg in self.messages if msg["role"] == "assistant")
        last_activity = self.messages[-1]["timestamp"] if self.messages else None
        
        return {
            "questions": questions,
            "responses": responses,
            "last_activity": last_activity,
            "status": self.status
        }

    @property
    def message_count(self) -> int:
        """Количество сообщений в чате"""
        return len(self.messages) if self.messages else 0

    @property
    def is_empty(self) -> bool:
        """Проверяет, пустой ли чат"""
        return self.message_count == 0

    @classmethod
    def get_or_create_for_user(cls, session, user_id: int) -> 'Chat':
        """Получает активный чат пользователя или создает новый"""
        chat = session.query(cls).filter(
            cls.user_id == user_id,
            cls.status == "active"
        ).first()
        
        if not chat:
            chat = cls(user_id=user_id, messages=[], status="active")
            session.add(chat)
            session.flush()
        
        return chat