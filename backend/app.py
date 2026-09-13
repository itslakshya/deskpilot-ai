"""
DeskPilot AI - FastAPI serving layer.

Endpoints:
  POST /predict   -> catalog category + priority recommendation, with confidence
                      and top-k alternatives (see src/inference.py for the logic)
  POST /feedback   -> accepts a user correction (e.g. "actually I meant X") and logs
                      it to feedback/queue.csv - this is the active-learning loop:
                      low-confidence + corrected predictions are the highest-value
                      examples to add to the next training batch.
  GET  /health     -> liveness/readiness probe + loaded model version metadata
  GET  /model-info -> current model version, training date, test metrics

Run with:  uvicorn app:app --reload --port 8000   (from the backend/ directory)
"""
import csv
import os
import sys
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from inference import DeskPilotService  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "..", "feedback", "queue.csv")

app = FastAPI(title="DeskPilot AI", version="1.0.0",
              description="Intent classification + priority recommendation for IT service desk tickets")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

service = None  # loaded once at startup (singleton) - avoids reloading model per request


@app.on_event("startup")
def load_model():
    global service
    t0 = time.time()
    service = DeskPilotService()
    print(f"Loaded models in {time.time() - t0:.2f}s")
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)


DEPARTMENTS = ["Finance", "HR", "Engineering", "Data Science", "Marketing", "Sales",
               "Legal", "Procurement", "Customer Support", "DevOps", "QA", "Design"]
ROLES = ["Associate", "Senior Associate", "Team Lead", "Manager", "Senior Manager", "Director", "VP"]


class PredictRequest(BaseModel):
    query_text: str = Field(..., min_length=3, max_length=500, description="Free-text request from the user")
    department: str = Field("Engineering", description="Requester's department")
    requester_role: str = Field("Associate", description="Requester's seniority/role")

    class Config:
        json_schema_extra = {"example": {
            "query_text": "need Adobe Photoshop installed on my laptop",
            "department": "Design", "requester_role": "Associate"}}


class Alternative(BaseModel):
    category: str
    confidence: float


class PredictResponse(BaseModel):
    request_id: str
    top_recommendation: str
    confidence: float
    low_confidence: bool
    alternatives: list[Alternative]
    fallback_message: str | None
    suggested_priority: str
    priority_confidence: float


class FeedbackRequest(BaseModel):
    request_id: str
    query_text: str
    shown_category: str
    correct_category: str
    was_correct: bool


@app.get("/")
def root():
    """The bare root previously 404'd, which reads as broken even though it isn't -
    this API has no UI of its own (see frontend/index.html for that), so point
    anyone who lands here at the interactive docs instead of a dead end."""
    return {
        "service": "DeskPilot AI",
        "message": "This is the JSON API, not the demo UI. Open frontend/index.html "
                    "directly in a browser for the interactive demo - it runs "
                    "entirely client-side and doesn't need this server.",
        "interactive_docs": "/docs",
        "endpoints": ["/predict", "/feedback", "/health", "/model-info"],
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi import Response
    return Response(status_code=204)  # silences the harmless browser favicon 404


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": service is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model-info")
def model_info():
    import json
    with open(f"{MODELS_DIR}/classical_results.json") as f:
        classical = json.load(f)
    return {
        "serving_model": "classical (Logistic Regression cascade)",
        "category_test_macro_f1": classical["category_model"]["test_macro_f1"],
        "priority_test_macro_f1": classical["priority_model"]["test_macro_f1"],
        "n_categories": 19, "n_priority_levels": 4,
        "trained_on_rows": classical["n_train"] + classical["n_val"] + classical["n_test"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if service is None:
        raise HTTPException(503, "Model not loaded yet")
    if req.department not in DEPARTMENTS:
        raise HTTPException(422, f"Unknown department '{req.department}'. Must be one of {DEPARTMENTS}")
    if req.requester_role not in ROLES:
        raise HTTPException(422, f"Unknown role '{req.requester_role}'. Must be one of {ROLES}")

    result = service.recommend(req.query_text, req.department, req.requester_role)
    result["request_id"] = str(uuid.uuid4())
    return result


@app.post("/feedback")
def feedback(fb: FeedbackRequest):
    """Logs a correction for the active-learning queue. Low-confidence + corrected
    examples get prioritized for human review before the next scripts/append_and_retrain.py run."""
    file_exists = os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["request_id", "query_text", "shown_category", "correct_category",
                              "was_correct", "logged_at"])
        writer.writerow([fb.request_id, fb.query_text, fb.shown_category, fb.correct_category,
                          fb.was_correct, datetime.now(timezone.utc).isoformat()])
    return {"status": "logged"}