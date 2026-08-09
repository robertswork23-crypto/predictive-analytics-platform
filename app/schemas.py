"""Pydantic request/response models."""
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from .ml import FEATURE_NAMES


class PredictRequest(BaseModel):
    features: List[float] = Field(min_length=len(FEATURE_NAMES), max_length=len(FEATURE_NAMES))


class PredictResponse(BaseModel):
    prediction: float
    model_name: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    r2: float
    mae: float
    rmse: float
    n_train_samples: int
    trained_at: datetime
    feature_names: List[str]


class ModelRunResponse(BaseModel):
    model_name: str
    r2: float
    mae: float
    rmse: float
    n_train_samples: int
    trained_at: datetime

    class Config:
        from_attributes = True


class PredictionLogResponse(BaseModel):
    features: List[float]
    prediction: float
    model_name: str
    created_at: datetime

    class Config:
        from_attributes = True
