from sqlalchemy import Column, Integer, Text, ForeignKey, DateTime, String, UniqueConstraint, func, Index
from sqlalchemy.orm import relationship
from app.db.base import Base

class BotCache(Base):
    __tablename__ = "bot_cache"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    question_hash = Column(String(64), nullable=False)
    answer = Column(Text, nullable=False)
    hit_count = Column(Integer, default=1, nullable=False)  # Счетчик использования
    cached_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_accessed = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="cache_entries")

    # Constraints и индексы
    __table_args__ = (
        UniqueConstraint("user_id", "question_hash", name="uq_user_question"),
        Index('idx_cache_user_accessed', 'user_id', 'last_accessed'),
        Index('idx_cache_accessed', 'last_accessed'),  # Для очистки старых записей
    )