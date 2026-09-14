"""
Single-command Runner for NeuroFuzzy Screening Web Server
Author: Soft Computing Project Team
"""

import sys
import os
import uvicorn

# Ensure backend root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

if __name__ == "__main__":
    print("=" * 65)
    print("  NEURO-FUZZY MENTAL HEALTH SCREENING PLATFORM")
    print("  Soft Computing University Course Project")
    print("=" * 65)
    print("  * Web Dashboard:  http://localhost:8000")
    print("  * API Docs:       http://localhost:8000/docs")
    print("=" * 65)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
