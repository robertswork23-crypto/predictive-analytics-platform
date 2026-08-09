"""Real model training + lightweight AutoML selection.

Loads scikit-learn's bundled diabetes regression dataset (real data, no
network call needed — it ships inside the sklearn package), trains a small
set of candidate regressors, picks the best by held-out R², and persists it
with joblib. No random numbers stand in for metrics here — every value
comes from an actual train/test split and an actual fit.
"""
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import config

logger = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
    "age", "sex", "bmi", "bp", "s1_tc", "s2_ldl", "s3_hdl", "s4_tch", "s5_ltg", "s6_glu",
]

CANDIDATE_MODELS = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "random_forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}


@dataclass
class TrainedModel:
    model_name: str
    estimator: object
    r2: float
    mae: float
    rmse: float
    n_train_samples: int
    trained_at: datetime


_lock = threading.Lock()
_current: Optional[TrainedModel] = None


def train_and_select() -> TrainedModel:
    """Fit every candidate on the same split, score on a real held-out set,
    keep the best by R². This is the "AutoML model selection" step — small
    and deterministic-ish rather than an exhaustive search, but genuinely
    comparing real fitted models, not picking a name and faking a score.
    """
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    best: Optional[TrainedModel] = None
    for name, estimator in CANDIDATE_MODELS.items():
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        logger.info(f"[automl] {name}: r2={r2:.4f} mae={mae:.2f} rmse={rmse:.2f}")

        if best is None or r2 > best.r2:
            best = TrainedModel(
                model_name=name,
                estimator=estimator,
                r2=r2,
                mae=mae,
                rmse=rmse,
                n_train_samples=len(X_train),
                trained_at=datetime.utcnow(),
            )

    logger.info(f"[automl] selected {best.model_name} (r2={best.r2:.4f})")
    joblib.dump({"model_name": best.model_name, "estimator": best.estimator}, config.model_path)

    with _lock:
        global _current
        _current = best

    return best


def get_current() -> TrainedModel:
    with _lock:
        if _current is None:
            raise RuntimeError("No model has been trained yet — call train_and_select() first")
        return _current


def predict(features: List[float]) -> Dict:
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features ({', '.join(FEATURE_NAMES)}), got {len(features)}")

    current = get_current()
    X = np.array(features).reshape(1, -1)
    prediction = float(current.estimator.predict(X)[0])
    return {"prediction": prediction, "model_name": current.model_name}


def sample_real_inputs(n: int = 1) -> List[List[float]]:
    """Real feature rows from the dataset itself, for a UI "try a real sample" button."""
    data = load_diabetes()
    rng = np.random.RandomState()
    idx = rng.choice(len(data.data), size=n, replace=False)
    return data.data[idx].tolist()
