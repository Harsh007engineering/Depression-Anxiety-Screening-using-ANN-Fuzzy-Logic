"""
Vercel Serverless Function Entry Point for FastAPI
NeuroFuzzy Clinical Screening Platform
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import Request
from backend.app.main import app

@app.middleware("http")
async def vercel_path_fix(request: Request, call_next):
    # Vercel rewrites sometimes pass /api/index.py as the path.
    # Check headers for the true client requested path.
    true_path = (
        request.headers.get("x-matched-path")
        or request.headers.get("x-forwarded-uri")
        or request.headers.get("x-original-uri")
        or request.headers.get("x-invoke-path")
    )
    if true_path and (request.scope.get("path") in ["/api/index.py", "/api/index", "/index.py"]):
        request.scope["path"] = true_path

    response = await call_next(request)
    return response

# Catch-all debug endpoint if /api/index.py is directly invoked
@app.api_route("/api/index.py", methods=["GET", "POST", "OPTIONS"])
@app.api_route("/index.py", methods=["GET", "POST", "OPTIONS"])
def debug_endpoint(request: Request):
    matched = request.headers.get("x-matched-path") or request.headers.get("x-forwarded-uri")
    return {
        "status": "online",
        "scope_path": request.scope.get("path"),
        "matched_path": matched,
        "headers": dict(request.headers)
    }

__all__ = ["app"]
