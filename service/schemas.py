"""Pydantic schemas for the service."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Incoming payload for overtime prediction."""

    fiscal_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    agency_name: Optional[str] = None
    title_description: Optional[str] = None
    work_location_borough: Optional[str] = None
    leave_status_as_of_june_30: Optional[str] = None
    pay_basis: Optional[str] = None
    base_salary: Optional[float] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class PredictionResponse(BaseModel):
    """Prediction output."""

    prob_ot: float
    pred_ot: int
    threshold: float
    model: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str