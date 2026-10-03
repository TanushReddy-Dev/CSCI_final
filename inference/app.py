"""
ConcreteGuard — Inference API Server
=====================================

FastAPI server providing the /predict endpoint.
Accepts image uploads, runs quality checks, inference, and Grad-CAM.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_DIR   — Path to model_artifacts/ directory (default: ../model_artifacts)
    EVAL_DIR    — Path to evaluation_outputs/ directory (default: ../evaluation_outputs)
    DEMO_MODE   — Set to "true" to use precomputed demo results
"""

import base64
import io
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Ensure inference directory is on path for submodule imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

# Configuration
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model_artifacts"))
EVAL_DIR = os.environ.get("EVAL_DIR", os.path.join(os.path.dirname(__file__), "..", "evaluation_outputs"))
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "gradcam_cache")

os.makedirs(GRADCAM_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ConcreteGuard API",
    description="Explainable concrete surface crack screening assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for Grad-CAM images
app.mount("/artifacts/gradcam", StaticFiles(directory=GRADCAM_DIR), name="gradcam")

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    label: str
    probability: float
    threshold: float
    visual_priority: str
    quality_status: str
    quality_message: str
    review_recommendation: str
    gradcam_url: Optional[str] = None
    gradcam_base64: Optional[str] = None
    limitations: str
    mode: str  # "live" or "demo"
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mode: str
    timestamp: str


class MetricsResponse(BaseModel):
    metrics: Optional[dict] = None
    hard_negative_summary: Optional[dict] = None
    error_case: Optional[dict] = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

predictor = None


def get_predictor():
    """Lazy-load the predictor."""
    global predictor
    if predictor is None and not DEMO_MODE:
        try:
            from predictor import CrackPredictor
            predictor = CrackPredictor(MODEL_DIR)
            print(f"[OK] Model loaded from {MODEL_DIR}")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")
            print("  Falling back to demo mode")
    return predictor


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

DEMO_RESULTS = {
    "crack_demo": {
        "label": "potential_crack",
        "probability": 0.92,
        "threshold": 0.60,
        "visual_priority": "high",
        "quality_status": "acceptable",
        "quality_message": "Image quality is acceptable for visual screening.",
        "review_recommendation": "Potential crack detected. Submit the image for expert inspection.",
        "limitations": "This is not a structural safety assessment. It detects visible crack-like image patterns only.",
    },
    "stain_demo": {
        "label": "no_visible_crack",
        "probability": 0.15,
        "threshold": 0.60,
        "visual_priority": "low",
        "quality_status": "acceptable",
        "quality_message": "Image quality is acceptable for visual screening.",
        "review_recommendation": "No visible crack detected in this image. This does not rule out hidden or structural damage.",
        "limitations": "This is not a structural safety assessment. It detects visible crack-like image patterns only.",
    },
    "shadow_demo": {
        "label": "uncertain_manual_review",
        "probability": 0.55,
        "threshold": 0.60,
        "visual_priority": "manual_review",
        "quality_status": "acceptable",
        "quality_message": "Image quality is acceptable for visual screening.",
        "review_recommendation": "Image quality or model confidence is insufficient. Manual review is recommended.",
        "limitations": "This is not a structural safety assessment. It detects visible crack-like image patterns only.",
    },
    "blurry_demo": {
        "label": "uncertain_manual_review",
        "probability": 0.40,
        "threshold": 0.60,
        "visual_priority": "manual_review",
        "quality_status": "warning",
        "quality_message": "Image appears blurry. Results may be less reliable.",
        "review_recommendation": "Image quality or model confidence is insufficient. Manual review is recommended.",
        "limitations": "This is not a structural safety assessment. It detects visible crack-like image patterns only.",
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pred = get_predictor()
    return HealthResponse(
        status="ok",
        model_loaded=pred is not None and pred.model is not None,
        mode="demo" if DEMO_MODE or pred is None else "live",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Analyze a concrete surface image for cracks.
    
    Returns prediction label, confidence, quality status, Grad-CAM evidence,
    and review recommendation.
    """
    from quality import run_all_checks

    # Read file
    file_bytes = await file.read()
    filename = file.filename or "upload.jpg"

    # Quality checks
    quality = run_all_checks(file_bytes, filename)

    if quality["status"] == "insufficient":
        return PredictionResponse(
            label="image_quality_insufficient",
            probability=0.0,
            threshold=0.0,
            visual_priority="manual_review",
            quality_status="insufficient",
            quality_message=quality["message"],
            review_recommendation="Image quality is insufficient for analysis. Please upload a clearer image.",
            limitations="This is not a structural safety assessment.",
            mode="live",
            timestamp=datetime.now().isoformat(),
        )

    image = quality["image"]
    pred = get_predictor()

    if pred is not None and pred.model is not None:
        # Live inference
        result = pred.predict_with_gradcam(image)

        # Save Grad-CAM to file
        gradcam_url = None
        if result.get("gradcam_base64"):
            gc_filename = f"{uuid.uuid4().hex}.png"
            gc_path = os.path.join(GRADCAM_DIR, gc_filename)
            gc_bytes = base64.b64decode(result["gradcam_base64"])
            with open(gc_path, "wb") as f:
                f.write(gc_bytes)
            gradcam_url = f"/artifacts/gradcam/{gc_filename}"

        return PredictionResponse(
            label=result["label"],
            probability=result["probability"],
            threshold=result["threshold"],
            visual_priority=result["visual_priority"],
            quality_status=quality["status"],
            quality_message=quality["message"],
            review_recommendation=result["review_recommendation"],
            gradcam_url=gradcam_url,
            gradcam_base64=result.get("gradcam_base64"),
            limitations=result["limitations"],
            mode="live",
            timestamp=datetime.now().isoformat(),
        )
    else:
        # Demo mode fallback
        demo = DEMO_RESULTS.get("crack_demo", {}).copy()
        demo["mode"] = "demo"
        demo["quality_status"] = quality["status"]
        demo["quality_message"] = quality["message"]
        demo["timestamp"] = datetime.now().isoformat()
        demo["gradcam_url"] = None
        demo["gradcam_base64"] = None
        return PredictionResponse(**demo)


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return evaluation metrics, hard-negative summary, and error case."""
    metrics = None
    hn_summary = None
    error_case = None

    metrics_path = os.path.join(EVAL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    hn_path = os.path.join(EVAL_DIR, "hard_negative_summary.json")
    if os.path.exists(hn_path):
        with open(hn_path) as f:
            hn_summary = json.load(f)

    error_path = os.path.join(EVAL_DIR, "error_case.json")
    if os.path.exists(error_path):
        with open(error_path) as f:
            error_case = json.load(f)

    return MetricsResponse(
        metrics=metrics,
        hard_negative_summary=hn_summary,
        error_case=error_case,
    )


@app.get("/demo-results")
async def demo_results():
    """Return all demo results for frontend demo mode."""
    return DEMO_RESULTS


@app.get("/")
async def root():
    """API root — redirect to docs."""
    return {
        "name": "ConcreteGuard API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "metrics": "GET /metrics",
            "health": "GET /health",
            "demo": "GET /demo-results",
        },
    }
