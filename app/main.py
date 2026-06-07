import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.preprocess import normalize_arabic_text

MODEL_PATH = Path("models/arabic_dialect_model.joblib")
FEEDBACK_PATH = Path("data/feedback/labeled_feedback.csv")
ALLOWED_LABELS = {"EGY", "GLF", "LAV", "MSA", "NOR"}

app = FastAPI(
    title="Arabic Dialect Classification API",
    description="Predicts Arabic dialect from a short Arabic text snippet.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount(
    "/resources", StaticFiles(directory="resources"), name="resources"
)  # for da cat image

model = None


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )

    model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    clean_text = normalize_arabic_text(request.text)

    if not clean_text:
        raise HTTPException(
            status_code=400, detail="Text is empty after preprocessing."
        )

    predicted_label = model.predict([clean_text])[0]

    if hasattr(model.named_steps["clf"], "predict_proba"):
        probabilities = model.predict_proba([clean_text])[0]
        classes = model.named_steps["clf"].classes_
        class_probs = {
            str(label): float(prob) for label, prob in zip(classes, probabilities)
        }
        probability = float(np.max(probabilities))
    else:
        class_probs = {str(predicted_label): 1.0}
        probability = 1.0

    return PredictionResponse(
        predicted_dialect=str(predicted_label),
        probability=probability,
        class_probabilities=class_probs,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def save_feedback(request: FeedbackRequest):
    text = normalize_arabic_text(request.text)
    correct_label = request.correct_dialect.strip().upper()
    predicted_label = request.predicted_dialect.strip().upper()

    if not text:
        raise HTTPException(
            status_code=400, detail="Text is empty after preprocessing."
        )

    if correct_label not in ALLOWED_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"correct_dialect must be one of: {sorted(ALLOWED_LABELS)}",
        )

    if predicted_label not in ALLOWED_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"predicted_dialect must be one of: {sorted(ALLOWED_LABELS)}",
        )

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_PATH.exists()

    with FEEDBACK_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "text",
                "predicted_label",
                "correct_label",
                "model_probability",
                "class_probabilities_json",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "predicted_label": predicted_label,
                "correct_label": correct_label,
                "model_probability": request.model_probability,
                "class_probabilities_json": json.dumps(
                    request.class_probabilities or {},
                    ensure_ascii=False,
                ),
            }
        )

    return FeedbackResponse(
        status="saved",
        message="Feedback saved for future retraining.",
        feedback_path=str(FEEDBACK_PATH),
    )
