from sqlmodel import SQLModel, Session, create_engine
from .config import settings

engine = create_engine(
    settings.app_db_url,
    connect_args={"check_same_thread": False} if settings.app_db_url.startswith("sqlite") else {}
)

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    return Session(engine)
