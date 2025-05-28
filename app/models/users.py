from sqlalchemy import Column, Integer, String, Boolean, DateTime, func, Index
from sqlalchemy.orm import relationship
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(320), unique=True, nullable=False)
    hashed_password = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    chats = relationship(
        "ChatHistory",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"  # Для больших объемов данных
    )
    
    cache_entries = relationship(
        "BotCache",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    # Индексы для производительности
    __table_args__ = (
        Index('idx_user_active_created', 'is_active', 'created_at'),
        Index('idx_user_email_active', 'email', 'is_active'),
    )