"""
FastAPI Main Application Entry Point
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .api.routes import router as api_router
from .core.config import FRONTEND_DIR, PROJECT_ROOT

app = FastAPI(
    title="NeuroFuzzy Mental Health Screening Engine",
    description="Soft Computing Hybrid System integrating Artificial Neural Networks (ANN) and Mamdani Fuzzy Inference System (FIS) for Explainable Depression & Anxiety Screening.",
    version="1.0.0"
)

# Enable CORS for local dev and decoupled web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes with /api prefix and without prefix for universal Vercel & Railway compatibility
app.include_router(api_router, prefix="/api", tags=["Screening & Soft Computing"])
app.include_router(api_router, tags=["Screening & Soft Computing (Direct)"])

# Candidate directories where frontend/static files might reside
candidate_dirs = [
    FRONTEND_DIR,
    os.path.join(PROJECT_ROOT, "frontend"),
    os.path.join(PROJECT_ROOT, "public"),
    os.path.join(os.getcwd(), "frontend"),
    os.path.join(os.getcwd(), "public"),
]

valid_frontend_dir = None
for cdir in candidate_dirs:
    if cdir and os.path.exists(cdir):
        valid_frontend_dir = cdir
        break

if valid_frontend_dir:
    app.mount("/static", StaticFiles(directory=valid_frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_index():
        index_path = os.path.join(valid_frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend index.html not found yet. Access /docs for API documentation."}
else:
    @app.get("/")
    def fallback_index():
        return {
            "status": "online",
            "service": "NeuroFuzzy Mental Health Screening Engine",
            "docs": "/docs"
        }
