"""ML Predictive Analytics Platform — FastAPI backend.

Real scikit-learn training with lightweight AutoML model selection over a
real dataset, real held-out metrics, a persisted model serving real
predictions, and a scheduled background retrain loop (APScheduler) standing
in for a separate Airflow deployment.
"""
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import schemas
from .config import config
from .db import ModelRun, PredictionLog, get_db, init_db
from .ml import get_current, predict as run_predict, sample_real_inputs, train_and_select
from .scheduler import start_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

init_db()

app = FastAPI(title="ML Predictive Analytics Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_request_counts = defaultdict(list)
RATE_LIMIT_PER_MINUTE = 100


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ("/api/health", "/docs", "/redoc", "/openapi.json"):
        return await call_next(request)

    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    now = datetime.utcnow()
    window_start = now - timedelta(minutes=1)
    _request_counts[ip] = [t for t in _request_counts[ip] if t > window_start]
    if len(_request_counts[ip]) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests")
    _request_counts[ip].append(now)

    return await call_next(request)


@app.on_event("startup")
async def on_startup():
    # Always train fresh at boot rather than loading a persisted estimator —
    # training is a fraction of a second on this dataset, and it guarantees
    # get_current() always reflects real, current held-out metrics instead of
    # a stale model.joblib left over from a previous deploy.
    result = train_and_select()
    db = next(get_db())
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

    start_scheduler()


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": config.environment,
    }


@app.get("/api/model/info", response_model=schemas.ModelInfoResponse)
async def model_info():
    from .ml import FEATURE_NAMES
    current = get_current()
    return schemas.ModelInfoResponse(
        model_name=current.model_name,
        r2=current.r2,
        mae=current.mae,
        rmse=current.rmse,
        n_train_samples=current.n_train_samples,
        trained_at=current.trained_at,
        feature_names=FEATURE_NAMES,
    )


@app.get("/api/model/sample")
async def model_sample():
    from .ml import FEATURE_NAMES
    features = sample_real_inputs(1)[0]
    return {"features": features, "feature_names": FEATURE_NAMES}


@app.post("/api/predict", response_model=schemas.PredictResponse)
async def predict(payload: schemas.PredictRequest, db: Session = Depends(get_db)):
    try:
        result = run_predict(payload.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    db.add(PredictionLog(
        features=payload.features,
        prediction=result["prediction"],
        model_name=result["model_name"],
    ))
    db.commit()

    return schemas.PredictResponse(
        prediction=result["prediction"],
        model_name=result["model_name"],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/api/predictions/recent", response_model=List[schemas.PredictionLogResponse])
async def recent_predictions(limit: int = 20, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    return db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit).all()


@app.post("/api/retrain", response_model=schemas.ModelRunResponse)
async def retrain(db: Session = Depends(get_db)):
    result = train_and_select()
    run = ModelRun(
        model_name=result.model_name,
        r2=result.r2,
        mae=result.mae,
        rmse=result.rmse,
        n_train_samples=result.n_train_samples,
        trained_at=result.trained_at,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@app.get("/api/model/history", response_model=List[schemas.ModelRunResponse])
async def model_history(limit: int = 20, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    return db.query(ModelRun).order_by(ModelRun.trained_at.desc()).limit(limit).all()


# ---- Frontend (built React SPA) — registered last so it never shadows /api/* ----
FRONTEND_DIST = (Path(__file__).parent.parent / "frontend_dist").resolve()

if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")


def _resolve_frontend_file(rel_path: str) -> Optional[Path]:
    rel_path = rel_path.lstrip("/\\")
    try:
        candidate = (FRONTEND_DIST / rel_path).resolve()
        candidate.relative_to(FRONTEND_DIST)
    except (ValueError, RuntimeError):
        return None
    return candidate


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    candidate = _resolve_frontend_file(full_path)
    if candidate is not None and candidate.is_file():
        return FileResponse(candidate)

    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)

    raise HTTPException(status_code=404, detail="Not found")
