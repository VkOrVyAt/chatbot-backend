from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from datetime import timedelta

from app.core.config import settings
from app.schemas.auth import UserLogin, Token, TokenData
from app.models.users import User
from app.core.utils.hash import verify_password
from app.core.utils.jwt import create_access_token


async def authenticate_user(user: UserLogin, db: AsyncSession) -> Token:
    """
    1) Принимаем UserLogin (username или email + password).
    2) Проверяем, что пользователь существует и пароль верный.
    3) Генерируем и возвращаем JWT (Token).
    """
    # Формируем условия для поиска пользователя
    conditions = []
    if user.username:
        conditions.append(User.username == user.username)
    if user.email:
        conditions.append(User.email == user.email)
    
    if not conditions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email must be provided",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Выполняем запрос к базе
    stmt = select(User).where(or_(*conditions))
    result = await db.execute(stmt)
    user_db = result.scalar_one_or_none()

    # Проверяем, что пользователь найден и пароль верный
    if not user_db or not verify_password(user.password, user_db.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or email",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Создаём объект TokenData с id пользователя
    token_data = TokenData(sub=str(user_db.id))
    
    # Генерируем JWT-токен
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(minutes=settings.JWT_EXPIRATION_TIME)
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.JWT_EXPIRATION_TIME * 60  # Convert minutes to seconds
    )