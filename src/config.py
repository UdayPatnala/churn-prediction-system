"""
Centralized configuration for the Churn Prediction System.

Keeps all file paths, feature definitions, and training hyperparameters
in a single location so nothing is hardcoded across modules.  Every value
falls back to a sensible default but can be overridden through environment
variables when deploying to different environments.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("CHURN_DATA_PATH", BASE_DIR / "data" / "customer_churn_sample.csv"))
ARTIFACT_DIR = Path(os.getenv("CHURN_ARTIFACT_DIR", BASE_DIR / "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURES_PATH = ARTIFACT_DIR / "features.json"

# ── Feature Definitions ─────────────────────────────────────────────────
NUMERIC_FEATURES: list[str] = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "support_tickets",
]

CATEGORICAL_FEATURES: list[str] = [
    "contract",
    "has_internet",
    "has_phone",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "churn"

# ── Training Hyperparameters ────────────────────────────────────────────
TEST_SIZE: float = float(os.getenv("CHURN_TEST_SIZE", "0.25"))
RANDOM_STATE: int = int(os.getenv("CHURN_RANDOM_STATE", "42"))
CV_FOLDS: int = int(os.getenv("CHURN_CV_FOLDS", "5"))

# XGBoost search space for GridSearchCV
XGB_PARAM_GRID: dict = {
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__n_estimators": [50, 100, 200],
}

# ── Allowed Categorical Values (used by Pydantic schemas) ───────────────
ALLOWED_CONTRACTS = ("Month-to-month", "One year", "Two year")
ALLOWED_YES_NO = ("Yes", "No")
