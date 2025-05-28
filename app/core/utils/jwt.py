from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError

from app.core.config import settings
from app.schemas.auth import TokenData

def create_access_token(data: TokenData, expires_delta: timedelta | None = None) -> str:
    # Преобразуем Pydantic-модель в словарь, исключая неустановленные поля
    to_encode = data.dict(exclude_unset=True)
    # Устанавливаем время истечения токена
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRATION_TIME)
    
    # Добавляем время истечения в словарь
    to_encode.update({"exp": int(expire.timestamp())})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def verify_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None
