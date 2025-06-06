from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from app.db.session import get_db
from app.schemas.user import UserCreate, UserRead, UserUpdate
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
    Registers a new user in the system.

    - **Rate Limit**: 5 requests per minute.
    - Creates a new user with a unique `username` and `email`.
    - The password is hashed before being stored in the database.

    Parameters:
    - **user** (UserCreate): The user data to register, including `username`, `email`, and `password`.
    - **db** (AsyncSession): The database session for creating the user.
    - **request** (Request, optional): The incoming HTTP request object (used for rate limiting).

    Returns:
    - **UserRead**: The created user's details, including `id`, `username`, `email`, `is_active`, and `created_at`.

    Raises:
    - **HTTPException** (400): If the `username` or `email` is already taken.
    - **HTTPException** (429): If the rate limit is exceeded.
    - **HTTPException** (500): If an unexpected server error occurs during registration.
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
    Authenticates a user and returns a JWT token.

    - **Rate Limit**: 10 requests per minute.
    - Validates the user's credentials and generates a JWT token.
    - Either `username` or `email` must be provided along with the `password`.

    Parameters:
    - **user** (UserLogin): The user credentials, including `username` (required), `email` (optional), and `password`.
    - **db** (AsyncSession): The database session for authenticating the user.
    - **request** (Request, optional): The incoming HTTP request object (used for rate limiting).

    Returns:
    - **Token**: The JWT token object containing `access_token`, `token_type` ("bearer"), and `expires_in` (in seconds).

    Raises:
    - **HTTPException** (400): If neither `username` nor `email` is provided.
    - **HTTPException** (401): If the credentials are invalid.
    - **HTTPException** (429): If the rate limit is exceeded.
    - **HTTPException** (500): If an unexpected server error occurs during authentication.
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