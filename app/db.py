"""SQLAlchemy engine/session + models. SQLite locally, Postgres (DATABASE_URL) in production.

Production Postgres is shared across multiple Nivor apps on the same Railway
project (cost-minimizing). Each app gets its own Postgres *schema* rather
than the default `public` schema so generic table names can't collide across
apps — this bit ai-chatbot-engine in production once (a shared `public.users`
table caused `create_all()` to silently skip creating this app's columns).
SQLite has no schema concept, so this only applies against Postgres/production.
"""
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import config

SCHEMA_NAME = "predictive_analytics"
_metadata = MetaData(schema=SCHEMA_NAME) if config.is_production else MetaData()
Base = declarative_base(metadata=_metadata)


class ModelRun(Base):
    """One training run's real held-out metrics — the "live dashboard" history."""

    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    r2 = Column(Float, nullable=False)
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    n_train_samples = Column(Integer, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow, index=True)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    features = Column(JSON, nullable=False)
    prediction = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def _database_url() -> str:
    if config.is_production:
        if not config.database_url:
            raise ValueError("DATABASE_URL is required in production")
        return config.database_url
    return config.database_url or "sqlite:///./predictive_analytics.db"


_url = _database_url()
_engine = create_engine(_url, connect_args={"check_same_thread": False} if "sqlite" in _url else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def init_db():
    if config.is_production:
        with _engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))
            conn.commit()
    Base.metadata.create_all(bind=_engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
