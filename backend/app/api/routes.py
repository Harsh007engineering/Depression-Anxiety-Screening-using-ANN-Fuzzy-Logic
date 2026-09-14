"""
API Route Handlers
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

import os
import json
import numpy as np
from fastapi import APIRouter, HTTPException
from ..core.schemas import ScreeningRequest, ScreeningResponse, QuestionItem
from ..core.config import MODEL_WEIGHTS_PATH, BENCHMARK_PATH
from ..core.constants import DASS21_QUESTIONS
from ..models.hybrid_pipeline import MentalHealthHybridPipeline
from ..models.fuzzy_engine import MentalHealthFuzzyEngine

router = APIRouter()

# Initialize singleton pipeline
pipeline = MentalHealthHybridPipeline(MODEL_WEIGHTS_PATH)
fuzzy_engine = MentalHealthFuzzyEngine()

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "NeuroFuzzy Mental Health Screening Engine",
        "ann_model_loaded": pipeline.ann.is_trained
    }

@router.get("/questions", response_model=list[QuestionItem])
def get_questions():
    """Returns the standardized 21 DASS items."""
    return DASS21_QUESTIONS

@router.post("/screen", response_model=ScreeningResponse)
def screen_mental_health(payload: ScreeningRequest):
    """
    Executes end-to-end Neuro-Fuzzy screening:
    ANN latent score extraction -> Mamdani FIS risk triaging & explainability.
    """
    try:
        result = pipeline.screen(
            responses_21=payload.responses,
            contextual_stress_factor=payload.contextual_stress
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fuzzy-visuals")
def get_fuzzy_visuals():
    """
    Computes coordinates for fuzzy membership functions across universes
    so the frontend can render dynamic interactive membership charts.
    """
    # 1. Depression / Anxiety / Stress (0 to 42)
    x_dass = np.linspace(0, 42, 85).tolist()
    
    def get_curve_points(var, x_points):
        curves = {tname: [] for tname in var.terms.keys()}
        for x in x_points:
            m = var.get_memberships(x)
            for tname, val in m.items():
                curves[tname].append(round(val, 3))
        return curves

    dep_curves = get_curve_points(fuzzy_engine.dep_var, x_dass)
    anx_curves = get_curve_points(fuzzy_engine.anx_var, x_dass)
    str_curves = get_curve_points(fuzzy_engine.str_var, x_dass)

    # 2. Risk Index (0 to 100)
    x_risk = np.linspace(0, 100, 101).tolist()
    risk_curves = get_curve_points(fuzzy_engine.risk_var, x_risk)

    return {
        "dass_x": [round(x, 1) for x in x_dass],
        "depression": dep_curves,
        "anxiety": anx_curves,
        "stress": str_curves,
        "risk_x": [round(x, 1) for x in x_risk],
        "risk": risk_curves
    }

@router.get("/benchmark")
def get_benchmark():
    """Returns comparative benchmark results between Pure ANN, Pure FIS, and Hybrid."""
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r") as f:
            return json.load(f)
    return {
        "metrics": [],
        "note": "Benchmark file not generated yet. Run backend/training/benchmark.py."
    }

@router.get("/personas")
def get_preset_personas():
    """Returns clinically calibrated pre-configured personas for live viva demonstrations."""
    return [
        {
            "id": "healthy",
            "name": "Alex — Resilient Baseline",
            "subtitle": "Healthy adaptive functioning with minimal situational distress",
            "contextual_stress": 0.5,
            "responses": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        },
        {
            "id": "exam_stress",
            "name": "Jordan — Exam Tension & Anxiety",
            "subtitle": "Noticeable situational pressure and moderate autonomic anxiety",
            "contextual_stress": 2.0,
            "responses": [1, 1, 0, 1, 0, 2, 1, 2, 1, 0, 1, 2, 0, 1, 1, 0, 0, 1, 1, 1, 0]
        },
        {
            "id": "clinical_depression",
            "name": "Taylor — Severe Clinical Depression",
            "subtitle": "Marked anhedonia, hopelessness, and inertia with moderate anxiety",
            "contextual_stress": 4.0,
            "responses": [1, 1, 2, 1, 2, 1, 0, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 0, 1, 0, 1]
        },
        {
            "id": "critical_crisis",
            "name": "Morgan — Acute Psychiatric Crisis",
            "subtitle": "Maximum acute clinical distress across depression, panic, and burnout",
            "contextual_stress": 9.0,
            "responses": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        }
    ]
