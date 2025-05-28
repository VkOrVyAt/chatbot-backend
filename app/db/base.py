from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

from app.models.users import User
from app.models.chats import ChatHistory
from app.models.bot_cashe import BotCache