from sqlalchemy import Column, Integer, Text, ForeignKey, DateTime, String, func, Index
from sqlalchemy.orm import relationship
from app.db.base import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    session_id = Column(String(36), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="chats")

    # Индексы для быстрых запросов
    __table_args__ = (
        Index('idx_chat_user_session', 'user_id', 'session_id'),
        Index('idx_chat_user_created', 'user_id', 'created_at'),
        Index('idx_chat_session_created', 'session_id', 'created_at'),
    )