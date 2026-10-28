"""
Vercel Serverless Function Entry Point for FastAPI
NeuroFuzzy Clinical Screening Platform
"""

import sys
import os
from urllib.parse import urlencode

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import Request
from backend.app.main import app

@app.middleware("http")
async def vercel_path_fix(request: Request, call_next):
    # Retrieve target path forwarded from vercel.json rewrite rule: ?path=/api/...
    path_override = request.query_params.get("path")
    if path_override:
        request.scope["path"] = path_override
        # Clean query parameters so endpoints receive only genuine client queries
        query_dict = dict(request.query_params)
        query_dict.pop("path", None)
        request.scope["query_string"] = urlencode(query_dict).encode("utf-8")

    response = await call_next(request)
    return response

__all__ = ["app"]
