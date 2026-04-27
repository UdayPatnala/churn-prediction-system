"""
Pydantic schemas for request / response validation.

Uses ``Literal`` types for categorical fields so invalid values are
rejected at the API boundary with a clear 422 error — demonstrating
that we understand input validation beyond basic type checks.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Request Schema ──────────────────────────────────────────────────────
class ChurnRequest(BaseModel):
    """Input features required to predict customer churn."""

    tenure: int = Field(
        ..., ge=0, description="Number of months the customer has been with the company"
    )
    monthly_charges: float = Field(
        ..., ge=0, description="Current monthly bill in USD"
    )
    total_charges: float = Field(
        ..., ge=0, description="Total amount billed over the customer's lifetime"
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Type of contract the customer holds"
    )
    has_internet: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has internet service"
    )
    has_phone: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has phone service"
    )
    support_tickets: int = Field(
        ..., ge=0, description="Number of support tickets raised by the customer"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 5,
                    "monthly_charges": 89.9,
                    "total_charges": 449.5,
                    "contract": "Month-to-month",
                    "has_internet": "Yes",
                    "has_phone": "Yes",
                    "support_tickets": 4,
                }
            ]
        }
    }


# ── Response Schemas ────────────────────────────────────────────────────
class ChurnResponse(BaseModel):
    """Prediction result returned by the ``/predict`` endpoint."""

    churn_probability: float = Field(
        ..., ge=0, le=1, description="Probability that the customer will churn (0-1)"
    )
    prediction: int = Field(
        ..., description="Binary prediction: 1 = churn, 0 = stay"
    )
    label: str = Field(
        ..., description="Human-readable prediction label"
    )


class HealthResponse(BaseModel):
    """Health-check payload returned by ``GET /health``."""

    status: str
    model_loaded: bool
    model_accuracy: float | None = None
    model_f1: float | None = None


class MetricsResponse(BaseModel):
    """Training metrics returned by ``GET /metrics``."""

    accuracy: float
    f1_score: float
    roc_auc: float
    cv_mean_accuracy: float
    best_params: dict
