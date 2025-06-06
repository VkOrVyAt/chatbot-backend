from sqlalchemy import Column, Integer, Text, String, DateTime, func, Index
from app.db.base import Base

class BotCache(Base):
    __tablename__ = "bot_cache"

    id = Column(Integer, primary_key=True, index=True)
    question_hash = Column(String(64), unique=True, nullable=False)
    answer = Column(Text, nullable=False)
    hit_count = Column(Integer, default=1, nullable=False)
    cached_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_accessed = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index('idx_cache_question_hash', 'question_hash'),
        Index('idx_cache_accessed', 'last_accessed'),
    )