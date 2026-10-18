"""Request/response schemas. Ranges follow the UCI codebook plus generous physiological limits."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """One patient. age/sex/cp are required; anything else may be omitted and is imputed."""

    model_config = ConfigDict(
        extra="forbid",  # reject unknown fields (typos, injection of unexpected keys)
        json_schema_extra={
            "example": {
                "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
                "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 2.3, "slope": 2,
            }
        },
    )

    age: int = Field(..., ge=18, le=110, description="years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float | None = Field(None, ge=60, le=260, description="resting blood pressure (mm Hg)")
    chol: float | None = Field(None, ge=80, le=700, description="serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] | None = Field(None, description="fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] | None = Field(None, description="resting ECG result")
    thalach: float | None = Field(None, ge=50, le=250, description="max heart rate achieved")
    exang: Literal[0, 1] | None = Field(None, description="exercise-induced angina")
    oldpeak: float | None = Field(None, ge=-5, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] | None = Field(None, description="slope of peak exercise ST segment")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: Literal[0, 1]
    label: Literal["no_disease", "disease"]
    confidence: float = Field(..., description="probability of the predicted class")
    probability_disease: float
    model_version: str
    request_id: str
