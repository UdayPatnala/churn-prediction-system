# Customer Churn Prediction System

A practical ML project that trains a churn classifier and serves predictions through FastAPI.

## Stack

- Python
- Scikit-learn
- FastAPI

## Run

```bash
pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
```

## Endpoints

- `GET /health`
- `POST /predict`
