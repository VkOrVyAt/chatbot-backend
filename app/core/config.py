from pydantic_settings import BaseSettings
from pydantic import AnyUrl
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

class Settings(BaseSettings):
    DATABASE_URL: AnyUrl
    SECRET_KEY: str
    JWT_EXPIRATION_TIME: int
    JWT_ALGORITHM: str
    REDIS_URL: str
    ENABLE_RATE_LIMITING: bool

    model_config = {
        "env_file": "app/.env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

settings = Settings()