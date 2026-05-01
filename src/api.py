"""
FastAPI application for serving real-time churn predictions.

Features:
    - Typed request/response models using Pydantic
    - Structured error handling (returns 422 for bad inputs, 500/404 for model issues)
    - Request ID tracking middleware for observability
    - Health endpoint that exposes model versioning/status
    - Metrics endpoint exposing training performance
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import METRICS_PATH, MODEL_PATH
from src.schemas import ChurnRequest, ChurnResponse, HealthResponse, MetricsResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Application Setup ───────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn API",
    description="Real-time telecom customer churn classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Middleware ──────────────────────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request: Request, call_next: Any) -> Any:
    """Inject a unique Request ID for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ── Model Management ────────────────────────────────────────────────────

_model = None
_metrics: dict[str, Any] | None = None

def _load_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load the trained model and metrics from disk lazily."""
    global _model, _metrics

    if _model is None:
        if not MODEL_PATH.exists():
            logger.error("Model file not found at %s", MODEL_PATH)
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please run the training pipeline first.",
            )
        try:
            logger.info("Loading model from %s", MODEL_PATH)
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            logger.exception("Failed to load model")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    if _metrics is None:
        if METRICS_PATH.exists():
            try:
                _metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Could not load metrics from %s: %s", METRICS_PATH, e)
                _metrics = {}
        else:
            _metrics = {}

    return _model, _metrics

# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root() -> dict[str, Any]:
    """Provide a simple landing response for API deployments."""
    return {
        "name": app.title,
        "status": "ok",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> dict[str, Any]:
    """Check API health and model availability."""
    try:
        _, metrics = _load_artifacts()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_accuracy": metrics.get("accuracy"),
            "model_f1": metrics.get("f1_score"),
        }
    except HTTPException:
        return {
            "status": "degraded",
            "model_loaded": False,
            "model_accuracy": None,
            "model_f1": None,
        }

@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def get_metrics() -> dict[str, Any]:
    """Retrieve performance metrics from the latest training run."""
    _, metrics = _load_artifacts()
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not available.")
    return metrics

@app.post("/predict", response_model=ChurnResponse, tags=["Prediction"])
def predict_churn(request: ChurnRequest, req: Request) -> dict[str, Any]:
    """Predict customer churn probability."""
    model, _ = _load_artifacts()

    # Convert Pydantic model to DataFrame (Scikit-learn pipeline expects DF)
    # The pipeline handles scaling and one-hot encoding internally.
    row = pd.DataFrame([request.model_dump()])

    try:
        # predict_proba returns [[prob_stay, prob_churn]]
        probability = float(model.predict_proba(row)[0][1])
        prediction = int(probability >= 0.5)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed during inference.")

    label = "Likely to churn" if prediction == 1 else "Likely to stay"

    logger.info(
        "ReqID:%s - Prediction: prob=%.4f label=%s inputs=%s",
        req.state.request_id,
        probability,
        label,
        request.model_dump(),
    )

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "label": label,
    }
