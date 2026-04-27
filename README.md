# Customer Churn Prediction — ML API

A machine learning pipeline that trains a customer churn classifier on telecom-style data and serves real-time predictions through a FastAPI REST endpoint.

## Tech Stack

- **Python 3.10+**
- **Scikit-learn** — Logistic Regression with preprocessing pipeline
- **FastAPI** — async prediction API
- **Pandas** — data handling

## Architecture

```
data/customer_churn_sample.csv
        │
        ▼
  ┌───────────┐      ┌────────────────────┐
  │  train.py  │─────▶│ artifacts/         │
  │            │      │  churn_model.joblib │
  └───────────┘      │  metrics.json       │
                      └────────┬───────────┘
                               │  loaded at startup
                               ▼
                      ┌────────────────┐
                      │   api.py       │
                      │  POST /predict │
                      │  GET  /health  │
                      └────────────────┘
```

## Setup & Run

```bash
pip install -r requirements.txt

# Train the model
python src/train.py

# Start the API server
uvicorn src.api:app --reload
```

## API Reference

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /predict`

**Request:**

```json
{
  "tenure": 5,
  "monthly_charges": 89.9,
  "total_charges": 449.5,
  "contract": "Month-to-month",
  "has_internet": "Yes",
  "has_phone": "Yes",
  "support_tickets": 4
}
```

**Response:**

```json
{
  "churn_probability": 0.8321,
  "prediction": 1,
  "label": "Likely to churn"
}
```

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `tenure` | int | Months as a customer |
| `monthly_charges` | float | Current monthly bill |
| `total_charges` | float | Lifetime spend |
| `contract` | str | Month-to-month / One year / Two year |
| `has_internet` | str | Yes / No |
| `has_phone` | str | Yes / No |
| `support_tickets` | int | Number of tickets raised |

## Project Structure

```
├── data/
│   └── customer_churn_sample.csv   # Training dataset (50 records)
├── src/
│   ├── __init__.py
│   ├── train.py                    # Model training + evaluation
│   ├── api.py                      # FastAPI prediction endpoint
│   └── schemas.py                  # Pydantic request schema
├── artifacts/                      # Trained model + metrics (git-ignored)
├── requirements.txt
└── README.md
```

## Key Concepts

- Scikit-learn Pipeline with ColumnTransformer (mixed numeric + categorical)
- Stratified train/test split for imbalanced classes
- Pydantic schema validation on API inputs
- Model serialization with joblib

## License

MIT
