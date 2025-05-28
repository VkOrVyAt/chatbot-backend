from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from app.routers.auth import router as auth_router
from app.core.config import setup_logging

# Настройка lifespan-обработчика
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения FastAPI.
    
    - При запуске: выполняет начальную настройку.
    - При остановке: выполняет очистку.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting ChatBot API")
    
    yield  # Здесь приложение работает
    
    logger.info("Shutting down ChatBot API")

# Инициализация приложения
app = FastAPI(
    title="ChatBot API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Настройка логирования (логи в logs/app.log в корне проекта)
setup_logging()

# Подключение роутеров
app.include_router(auth_router)

@app.get("/")
async def root():
    """
    Корневой эндпоинт для проверки работоспособности API.
    """
    return {"message": "API is up and running 🚀"}