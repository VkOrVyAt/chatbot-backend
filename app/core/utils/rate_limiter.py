from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
import redis
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

def get_client_ip(request: Request) -> str:
    """Получает IP клиента с учетом прокси-серверов"""
    # Проверяем заголовки прокси (порядок важен!)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Cloudflare
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip
    
    # Fallback на стандартную функцию slowapi
    return get_remote_address(request)

def create_limiter() -> Limiter:
    """Создает и настраивает rate limiter"""
    try:
        # Проверяем подключение к Redis
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        redis_client.ping()
        
        limiter = Limiter(
            key_func=get_client_ip,
            storage_uri=settings.REDIS_URL,
            default_limits=["200/minute", "20/second"]  # Более гибкие лимиты
        )
        logger.info("✅ Rate limiter initialized with Redis storage")
        return limiter
        
    except Exception as e:
        logger.warning(f"⚠️ Redis unavailable, using in-memory storage: {e}")
        
        # Fallback на in-memory для разработки
        limiter = Limiter(
            key_func=get_client_ip,
            default_limits=["200/minute", "20/second"]
        )
        logger.info("✅ Rate limiter initialized with in-memory storage")
        return limiter
    
limiter = create_limiter()
# Добавляем обработчик исключений для превышения лимита