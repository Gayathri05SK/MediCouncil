from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class TriageRequest(BaseModel):
    symptoms_text: str = Field(..., min_length=3, max_length=2000, description="Patient symptom description")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age")
    sex: Optional[str] = Field(None, description="Patient sex (M/F/U)")
    chronic_conditions: Optional[List[str]] = Field(default=[], description="List of chronic conditions")
    red_flags: Optional[List[str]] = Field(default=[], description="List of red flag symptoms")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptoms_text": "severe chest pain radiating to left arm with sweating",
                "age": 55,
                "sex": "M",
                "chronic_conditions": ["hypertension", "diabetes"],
                "red_flags": ["chest_pain", "difficulty_breathing"]
            }
        }

class TriageResponse(BaseModel):
    status: str
    risk_level: Optional[str] = None
    risk_score: Optional[int] = None
    urgency: Optional[str] = None
    confidence: Optional[float] = None
    confidence_level: Optional[str] = None
    explanation: Optional[str] = None
    key_symptoms: Optional[List[str]] = None
    agent_agreement: Optional[Dict] = None
    safety_override: Optional[bool] = None
    disclaimer: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
