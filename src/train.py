from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/customer_churn_sample.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "features.json"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    feature_cols = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract",
        "has_internet",
        "has_phone",
        "support_tickets",
    ]

    target_col = "churn"

    X = df[feature_cols]
    y = df[target_col]

    numeric = ["tenure", "monthly_charges", "total_charges", "support_tickets"]
    categorical = ["contract", "has_internet", "has_phone"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("F1 Score:", round(f1_score(y_test, preds), 4))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
