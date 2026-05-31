"""Database utilities."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from outbound_eval.infra.config import settings

Base = declarative_base()

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database."""
    Base.metadata.create_all(bind=engine)