from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from app.db.session import get_db
from app.schemas.user import UserCreate, UserRead
from app.schemas.auth import UserLogin, Token
from app.services.registration import register_user
from app.services.auth import authenticate_user
from app.core.utils.rate_limiter import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED, summary="Register a new user")
@limiter.limit("5/minute")
async def register(user: UserCreate, db: AsyncSession = Depends(get_db), request: Request = None):
    """
    Эндпоинт для регистрации нового пользователя.

    - Принимает данные пользователя (username, email, password) в формате UserCreate.
    - Проверяет уникальность username и email.
    - Хеширует пароль и сохраняет пользователя в базе данных.
    - Возвращает данные созданного пользователя (id, username, email, is_active, created_at).

    Args:
        user (UserCreate): Данные для создания пользователя.
        db (AsyncSession): Асинхронная сессия базы данных.

    Returns:
        UserRead: Данные созданного пользователя.

    Raises:
        HTTPException: Если username или email уже заняты (400) или произошла ошибка сервера (500).
    """
    try:
        new_user = await register_user(user, db)
        logger.info(f"User registered: {user.username}")
        return new_user
    except HTTPException as e:
        logger.warning(f"Registration failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )
    
@router.post("/login", response_model=Token, summary="Authenticate a user")
@limiter.limit("10/minute")
async def login(user: UserLogin, db: AsyncSession = Depends(get_db), request: Request = None):
    """
    Эндпоинт для авторизации пользователя.

    - Принимает username (обязательно), email (опционально) и password в формате UserLogin.
    - Проверяет существование пользователя и правильность пароля.
    - Генерирует и возвращает JWT-токен с id пользователя в поле sub.

    Args:
        user (UserLogin): Данные для входа.
        db (AsyncSession): Асинхронная сессия базы данных.

    Returns:
        Token: Объект с access_token, token_type ("bearer") и expires_in (в секундах).

    Raises:
        HTTPException: Если данные неверны (401), не предоставлены username/email (400) или произошла ошибка сервера (500).
    """
    try:
        token = await authenticate_user(user, db)
        logger.info(f"User logged in: {user.username}")
        return token
    except HTTPException as e:
        logger.warning(f"Login failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to authenticate user"
        )
