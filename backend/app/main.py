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
from .core.config import FRONTEND_DIR

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

# Register API routes
app.include_router(api_router, prefix="/api", tags=["Screening & Soft Computing"])

# Mount Frontend static assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_frontend_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend index.html not found yet. Access /docs for API documentation."}
