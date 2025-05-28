from passlib.context import CryptContext
# Create a password context for hashing and verifying passwords

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """
    Хеширует пароль пользователя
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Проверяет, что незахешированный пароль соответствует хранившемуся хешу.
    """
    return pwd_context.verify(plain_password, hashed_password)