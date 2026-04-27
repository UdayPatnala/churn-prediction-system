from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import ChurnRequest

MODEL_PATH = Path("artifacts/churn_model.joblib")

app = FastAPI(title="Churn Prediction API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ChurnRequest) -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=400, detail="Model not found. Run `python src/train.py` first.")

    model = joblib.load(MODEL_PATH)

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

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "label": "Likely to churn" if prediction == 1 else "Likely to stay",
    }
