"""Background scheduled retraining — the practical substitute for a separate
Airflow deployment: a real interval-based retrain loop running in-process,
logging a real ModelRun row on every cycle.
"""
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .config import config
from .db import ModelRun, SessionLocal
from .ml import train_and_select

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


def _retrain_job():
    logger.info("[scheduler] running scheduled retrain")
    result = train_and_select()
    db = SessionLocal()
    try:
        db.add(ModelRun(
            model_name=result.model_name,
            r2=result.r2,
            mae=result.mae,
            rmse=result.rmse,
            n_train_samples=result.n_train_samples,
            trained_at=result.trained_at,
        ))
        db.commit()
    finally:
        db.close()


def start_scheduler():
    scheduler.add_job(
        _retrain_job,
        "interval",
        minutes=config.retrain_interval_minutes,
        id="retrain",
        replace_existing=True,
    )
    scheduler.start()
