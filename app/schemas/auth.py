from pydantic import BaseModel
from typing import Optional

class UserLogin(BaseModel):
    username: str
    email: Optional[str] = None
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    sub: Optional[str] = None