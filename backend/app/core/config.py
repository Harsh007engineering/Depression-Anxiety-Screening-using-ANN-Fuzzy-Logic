"""
Application Configuration
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

import os

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CORE_DIR)
BACKEND_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODEL_WEIGHTS_PATH = os.path.join(APP_DIR, "models", "saved", "mental_health_ann.joblib")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
BENCHMARK_PATH = os.path.join(BACKEND_DIR, "training", "benchmark_results.json")

if __name__ == "__main__":
    print("CORE_DIR:          ", CORE_DIR)
    print("APP_DIR:           ", APP_DIR)
    print("BACKEND_DIR:       ", BACKEND_DIR)
    print("PROJECT_ROOT:      ", PROJECT_ROOT)
    print("MODEL_WEIGHTS_PATH:", MODEL_WEIGHTS_PATH, "Exists?", os.path.exists(MODEL_WEIGHTS_PATH))
    print("FRONTEND_DIR:      ", FRONTEND_DIR, "Exists?", os.path.exists(FRONTEND_DIR))
    print("BENCHMARK_PATH:    ", BENCHMARK_PATH, "Exists?", os.path.exists(BENCHMARK_PATH))
