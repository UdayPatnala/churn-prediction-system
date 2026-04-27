"""
Model training pipeline for Customer Churn Prediction.

Pipeline stages:
    1. Load & validate the CSV dataset (nulls, duplicates, class balance).
    2. Build a Scikit-learn ``Pipeline`` with ``ColumnTransformer`` to handle
       mixed numeric (``StandardScaler``) and categorical (``OneHotEncoder``)
       features in a single, serializable object.
    3. Perform stratified train / test split to preserve class ratios.
    4. Run ``GridSearchCV`` to tune Logistic Regression hyper-parameters
       (``C`` and ``solver``) with k-fold cross-validation.
    5. Evaluate on the hold-out set: accuracy, F1, ROC-AUC, confusion matrix.
    6. Persist the best model (``joblib``) and all metrics (``JSON``)
       to the ``artifacts/`` directory.

Design choice — *Logistic Regression over XGBoost*:
    For a telecom churn dataset with < 1 000 records and 7 features,
    Logistic Regression offers comparable accuracy with full coefficient
    interpretability, which is valuable when presenting results to
    non-technical stakeholders.  The pipeline is structured so swapping
    in ``XGBClassifier`` or ``RandomForestClassifier`` requires changing
    only the ``classifier`` step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_PATH,
    FEATURE_COLUMNS,
    FEATURES_PATH,
    XGB_PARAM_GRID,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Data Validation ─────────────────────────────────────────────────────

def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run basic sanity checks on the raw dataset before training.

    Checks for:
        - Missing values (drops rows with NaNs and warns).
        - Duplicate rows (drops and warns).
        - Class distribution of the target column.
    """
    logger.info("── Data Validation ──")

    # Missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning("Found %d null value(s):\n%s", total_nulls, null_counts[null_counts > 0])
        df = df.dropna()
        logger.info("Dropped rows with nulls → new shape: %s", df.shape)
    else:
        logger.info("No missing values detected ✓")

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.warning("Found %d duplicate row(s) — dropping.", dup_count)
        df = df.drop_duplicates()
    else:
        logger.info("No duplicate rows ✓")

    # Class balance
    class_dist = df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    logger.info(
        "Class distribution → %s",
        {k: f"{v:.1%}" for k, v in class_dist.items()},
    )

    # Basic statistics for numeric columns
    logger.info("Numeric summary:\n%s", df[NUMERIC_FEATURES].describe().round(2).to_string())

    return df


# ── Pipeline Construction ────────────────────────────────────────────────

def _build_pipeline() -> Pipeline:
    """Create a Scikit-learn Pipeline with ColumnTransformer preprocessing.

    The ``ColumnTransformer`` applies:
        - ``StandardScaler`` to numeric features (zero-mean, unit-variance)
        - ``OneHotEncoder`` to categorical features (``handle_unknown='ignore'``
          ensures unseen categories at prediction time don't crash the model)

    Returns:
        An unfitted ``Pipeline`` ready for ``GridSearchCV``.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")),
        ]
    )


# ── Training Entry Point ────────────────────────────────────────────────

def main() -> None:
    """Execute the full training pipeline and persist artifacts."""

    # 1. Load data
    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Raw dataset shape: %s", df.shape)

    # 2. Validate
    df = _validate_dataframe(df)

    # 3. Split features / target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train set: %d samples | Test set: %d samples", len(X_train), len(X_test))

    # 4. Cross-validated hyperparameter search
    pipeline = _build_pipeline()

    logger.info("Running GridSearchCV with %d-fold CV …", CV_FOLDS)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=XGB_PARAM_GRID,
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info("Best parameters: %s", grid_search.best_params_)
    logger.info("Best CV F1: %.4f", grid_search.best_score_)

    # 5. Cross-validation score on full training set (for reporting)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy")
    logger.info("CV Accuracy: %.4f (± %.4f)", cv_scores.mean(), cv_scores.std())

    # 6. Evaluate on hold-out test set
    preds = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = round(accuracy_score(y_test, preds), 4)
    f1 = round(f1_score(y_test, preds), 4)
    roc_auc = round(roc_auc_score(y_test, proba), 4)
    cm = confusion_matrix(y_test, preds).tolist()

    logger.info("── Hold-out Test Results ──")
    logger.info("Accuracy : %s", accuracy)
    logger.info("F1 Score : %s", f1)
    logger.info("ROC-AUC  : %s", roc_auc)
    logger.info("Confusion Matrix:\n%s", np.array(cm))
    logger.info("\n%s", classification_report(y_test, preds, target_names=["Stay", "Churn"]))

    # 7. Persist artifacts
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
        "confusion_matrix": cm,
        "best_params": grid_search.best_params_,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Model saved  → %s", MODEL_PATH)
    logger.info("Metrics saved → %s", METRICS_PATH)
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
