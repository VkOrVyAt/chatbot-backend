from sqlalchemy import Column, Integer, Text, String, DateTime, func, Index
import hashlib
from typing import Optional, List, Dict
from datetime import datetime
from app.db.base import Base

class BotCache(Base):
    __tablename__ = "bot_cache"

    id = Column(Integer, primary_key=True, index=True)
    question_hash = Column(String(64), unique=True, nullable=False)
    question_text = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(String(50), nullable=True)
    hit_count = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_used_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index('idx_cache_question_hash', 'question_hash'),
        Index('idx_cache_category', 'category'),
        Index('idx_cache_hit_count', 'hit_count'),
    )

    @staticmethod
    def hash_question(question: str) -> str:
        """Создает хэш вопроса для кеширования"""
        normalized = " ".join(question.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def increment_hit(self) -> None:
        """Увеличивает счетчик попаданий в кеш"""
        self.hit_count += 1

    @classmethod
    def get_cached_answer(cls, session, question: str) -> Optional[str]:
        """Ищет ответ в кеше"""
        question_hash = cls.hash_question(question)
        cache_entry = session.query(cls).filter(cls.question_hash == question_hash).first()
        
        if cache_entry:
            cache_entry.increment_hit()
            return cache_entry.answer
        
        return None

    @classmethod
    def save_answer(cls, session, question: str, answer: str, category: str = None) -> None:
        """Сохраняет ответ в кеш"""
        question_hash = cls.hash_question(question)
        
        existing = session.query(cls).filter(cls.question_hash == question_hash).first()
        if existing:
            existing.answer = answer
            existing.category = category
            existing.increment_hit()
        else:
            cache_entry = cls(
                question_hash=question_hash,
                question_text=question[:500],
                answer=answer,
                category=category
            )
            session.add(cache_entry)

    @classmethod
    def get_popular_questions(cls, session, limit: int = 10) -> List[Dict]:
        """Возвращает популярные вопросы"""
        results = (
            session.query(cls.question_text, cls.hit_count, cls.category)
            .order_by(cls.hit_count.desc())
            .limit(limit)
            .all()
        )
        
        return [
            {
                "question": result[0],
                "hits": result[1],
                "category": result[2]
            }
            for result in results
        ]

    @classmethod
    def cleanup_unused_cache(cls, session, min_hits: int = 2, days_old: int = 30) -> int:
        """Удаляет неиспользуемые записи кеша"""
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        deleted = session.query(cls).filter(
            cls.hit_count < min_hits,
            cls.last_used_at < cutoff_date
        ).delete()
        
        return deleted