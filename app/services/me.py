from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.users import User
from app.schemas.user import UserUpdate, UserRead

class MeService:
    @staticmethod
    async def update_current_user(user_id: int, user_update: UserUpdate, db: AsyncSession) -> User:
        """Uupdate the current user's profile"""
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        update_data = user_update.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        try:
            await db.commit()
            await db.refresh(user)
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user"
            )
        
        return user
    
    @staticmethod
    async def get_current_user_profile(user: User) -> UserRead:
        """Get the current user's profile information"""
        return UserRead.model_validate(user)