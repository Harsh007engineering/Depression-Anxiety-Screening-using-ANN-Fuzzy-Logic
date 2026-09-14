"""
Pydantic Schemas for Request & Response Validation
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

class ScreeningRequest(BaseModel):
    responses: List[int] = Field(
        ...,
        description="List of 21 integer responses corresponding to DASS-21 questions (values 0, 1, 2, or 3)"
    )
    contextual_stress: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Self-reported environmental or lifestyle stress rating (scale 0-10)"
    )

    @field_validator("responses")
    @classmethod
    def validate_responses(cls, v):
        if len(v) != 21:
            raise ValueError(f"Expected exactly 21 responses, got {len(v)}")
        for idx, val in enumerate(v):
            if val not in [0, 1, 2, 3]:
                raise ValueError(f"Response at index {idx} must be 0, 1, 2, or 3 (got {val})")
        return v

class QuestionItem(BaseModel):
    id: int
    subscale: str
    text: str

class FiredRule(BaseModel):
    rule_id: int
    description: str
    consequent: str
    weight: float

class ActionPlan(BaseModel):
    title: str
    recommendation: str
    urgency: str
    next_steps: List[str]

class SubscaleANN(BaseModel):
    score: float
    class_id: Optional[int] = None
    label: str
    probabilities: Dict[str, float]

class ScreeningResponse(BaseModel):
    dass21_raw_scores: Dict[str, int]
    ann_projections: Dict[str, Any]
    fuzzy_triage: Dict[str, Any]
    summary: Dict[str, Any]
