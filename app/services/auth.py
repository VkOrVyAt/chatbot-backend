from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from datetime import timedelta
import logging

from app.core.config import settings
from app.schemas.auth import UserLogin, Token, TokenData
from app.models.users import User
from app.core.utils.hash import verify_password
from app.core.utils.jwt import create_access_token

logger = logging.getLogger(__name__)

async def authenticate_user(user: UserLogin, db: AsyncSession) -> Token:
    """
    1) Принимаем UserLogin (username или email + password).
    2) Проверяем, что пользователь существует и пароль верный.
    3) Генерируем и возвращаем JWT (Token).
    """
    conditions = []
    if user.username:
        conditions.append(User.username == user.username)
    if user.email:
        conditions.append(User.email == user.email)
    
    if not conditions:
        logger.warning("Login attempt with no username or email provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email must be provided",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    stmt = select(User).where(or_(*conditions))
    result = await db.execute(stmt)
    user_db = result.scalar_one_or_none()

    if not user_db or not verify_password(user.password, user_db.hashed_password):
        logger.warning(f"Failed login attempt for username: {user.username or 'unknown'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or email",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = TokenData(sub=str(user_db.id))
    
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(minutes=settings.JWT_EXPIRATION_TIME)
    )

    logger.info(f"User logged in: {user_db.username}")
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.JWT_EXPIRATION_TIME * 60
    )