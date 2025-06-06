from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.utils.dependencies import get_current_active_user
from app.core.utils.rate_limiter import limiter
from app.schemas.user import UserRead, UserUpdate
from app.services.me import MeService
from app.models.users import User

router = APIRouter(prefix="/me", tags=["me"])

@router.get("", response_model=UserRead, summary="Get Current User Profile")
@limiter.limit("10/minute")
async def get_current_user_info(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """
    Retrieves the profile information of the currently authenticated user.

    - **Rate Limit**: 10 requests per minute.
    - Requires a valid JWT token in the `Authorization` header (Bearer scheme).

    Parameters:
    - **request** (Request): The incoming HTTP request object (used for rate limiting).
    - **current_user** (User): The authenticated user, extracted from the JWT token.

    Returns:
    - **UserRead**: The profile details of the current user, including `id`, `username`, `email`, `is_active`, and `created_at`.

    Raises:
    - **HTTPException** (401): If the JWT token is invalid or the user is not authenticated.
    """
    return await MeService.get_current_user_profile(current_user)

@router.put("", response_model=UserRead)
@limiter.limit("5/minute")
async def update_current_user(
    request: Request,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Updates the profile information of the currently authenticated user.

    - **Rate Limit**: 5 requests per minute.
    - Requires a valid JWT token in the `Authorization` header (Bearer scheme).
    - Only fields provided in the request body will be updated (partial update).

    Parameters:
    - **request** (Request): The incoming HTTP request object (used for rate limiting).
    - **user_update** (UserUpdate): The data to update the user profile (e.g., `username`, `email`).
    - **current_user** (User): The authenticated user, extracted from the JWT token.
    - **db** (AsyncSession): The database session for performing the update operation.

    Returns:
    - **UserRead**: The updated profile details of the current user, including `id`, `username`, `email`, `is_active`, and `created_at`.

    Raises:
    - **HTTPException** (401): If the JWT token is invalid or the user is not authenticated.
    - **HTTPException** (404): If the user is not found in the database.
    - **HTTPException** (500): If an unexpected server error occurs during the update.
    """
    updated_user = await MeService.update_current_user(current_user.id, user_update, db)
    return UserRead.model_validate(updated_user)