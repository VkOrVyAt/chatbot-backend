from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    username: str
    email: Optional[EmailStr] = None

class UserCreate(UserBase):
    password: str
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('Пароль должен содержать минимум 8 символов')
        if not any(c.isdigit() for c in v):
            raise ValueError('Пароль должен содержать хотя бы одну цифру')
        return v

class UserRead(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    model_config = {"from_attributes": True}

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None