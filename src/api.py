from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChurnRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = Path("artifacts/churn_model.joblib")

app = FastAPI(title="Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def _get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=400, detail="Model not found. Run `python src/train.py` first.")
        logger.info("Loading model from %s", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ChurnRequest) -> dict:
    model = _get_model()

    row = pd.DataFrame([
        {
            "tenure": request.tenure,
            "monthly_charges": request.monthly_charges,
            "total_charges": request.total_charges,
            "contract": request.contract,
            "has_internet": request.has_internet,
            "has_phone": request.has_phone,
            "support_tickets": request.support_tickets,
        }
    ])

    probability = float(model.predict_proba(row)[0][1])
    prediction = int(probability >= 0.5)

    logger.info("Prediction: prob=%.4f label=%s", probability, "churn" if prediction else "stay")

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "label": "Likely to churn" if prediction == 1 else "Likely to stay",
    }
