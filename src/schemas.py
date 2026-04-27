from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    tenure: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract: str
    has_internet: str
    has_phone: str
    support_tickets: int = Field(..., ge=0)
