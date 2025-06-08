from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

def setup_logging(log_level: str = "INFO"):
    """Настраивает логирование с ротацией по времени в папке logs корневого проекта."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "bot_service.log")

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(threadName)s - %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level.upper()))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[file_handler, console_handler]
    )
    logging.getLogger("bot_service").info("Logging initialized successfully")

class Settings(BaseSettings):
    DATABASE_URL: str
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    REDIS_URL: str
    MODEL_PATH: str

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file="bot_service/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    def __init__(self, *args, **kwargs):
        """Инициализация с настройкой логирования."""
        super().__init__(*args, **kwargs)
        setup_logging(self.LOG_LEVEL)

settings = Settings()