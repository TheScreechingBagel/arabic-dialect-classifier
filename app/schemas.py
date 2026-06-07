from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        examples=["انا عايز اعرف الاخبار النهارده"],
        description="Arabic text snippet to classify.",
    )


class PredictionResponse(BaseModel):
    predicted_dialect: str = Field(..., examples=["EGY"])
    probability: float = Field(..., examples=[0.82])
    class_probabilities: Dict[str, float] = Field(
        ...,
        examples=[
            {
                "EGY": 0.82,
                "GLF": 0.04,
                "LAV": 0.05,
                "MSA": 0.06,
                "NOR": 0.03,
            }
        ],
    )


class FeedbackRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        examples=["شو بدك تعمل هلق"],
        description="Arabic text snippet that was predicted by the model.",
    )
    predicted_dialect: str = Field(
        ...,
        examples=["GLF"],
        description="Dialect predicted by the current model.",
    )
    correct_dialect: str = Field(
        ...,
        examples=["LAV"],
        description="Human-corrected dialect label.",
    )
    model_probability: Optional[float] = Field(
        default=None,
        examples=[0.61],
        description="Probability assigned to the predicted class by the model.",
    )
    class_probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        examples=[
            {
                "EGY": 0.05,
                "GLF": 0.61,
                "LAV": 0.24,
                "MSA": 0.04,
                "NOR": 0.06,
            }
        ],
        description="Full class probability breakdown returned by the model.",
    )


class FeedbackResponse(BaseModel):
    status: str = Field(..., examples=["saved"])
    message: str = Field(..., examples=["Feedback saved for future retraining."])
    feedback_path: str = Field(..., examples=["data/feedback/labeled_feedback.csv"])
