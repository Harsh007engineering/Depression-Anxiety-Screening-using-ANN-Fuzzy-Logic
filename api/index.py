"""
Vercel Serverless Function Entry Point for FastAPI
NeuroFuzzy Clinical Screening Platform
"""

import sys
import os

# Add repository root directory to sys.path so 'backend' package is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.main import app

# Export the ASGI app instance for Vercel
__all__ = ["app"]
