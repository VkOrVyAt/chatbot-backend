from fastapi import HTTPException, status
from sqlalchemy import select, or_, exists
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.schemas.user import UserCreate, UserRead
from app.models.users import User
from app.core.utils.hash import get_password_hash

logger = logging.getLogger(__name__)

async def register_user(user_data: UserCreate, db: AsyncSession) -> UserRead:
    # 1. Проверяем, что такого username/email ещё нет
    stmt = select(exists().where(
        or_(
            User.email == user_data.email,
            User.username == user_data.username
        )
    ))
    exists_in_db = (await db.execute(stmt)).scalar_one()
    if exists_in_db:
        logger.warning(f"Registration failed: username {user_data.username} or email {user_data.email} already exists")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists."
        )

    # 2. Хешируем пароль
    hashed = get_password_hash(user_data.password)

    # 3. Создаём объект и сохраняем
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed,
    )
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
        logger.info(f"User registered: {new_user.username}")
    except IntegrityError:
        # Вдруг кто-то успел вставить того же пользователя параллельно
        await db.rollback()
        logger.error(f"Database integrity error during registration for {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not create user due to a database integrity error."
        )

    # 4. Возвращаем DTO
    # Pydantic v2: model_validate с конфигом from_attributes=True
    return UserRead.model_validate(new_user)