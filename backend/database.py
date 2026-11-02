import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from state import BASE_DIR

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    db_path = BASE_DIR / "sonicmap.db"
    DATABASE_URL = f"sqlite:///{db_path}"

# Render provides postgres:// which SQLAlchemy >= 1.4 requires as postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    import models  # Ensure all models are imported so metadata knows them
    Base.metadata.create_all(bind=engine)
