"""
ANN Training and Evaluation Script for Soft Computing Project
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Ensure backend root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from training.dataset_loader import generate_clinically_calibrated_dataset, DASS21_QUESTIONS
from app.models.ann_model import MentalHealthANN

def train_and_evaluate():
    print("=" * 60)
    print("Step 1: Generating Clinically Calibrated DASS-21 Dataset (6000 samples)...")
    df = generate_clinically_calibrated_dataset(n_samples=6000, random_state=42)
    
    q_cols = [f"Q{q['id']}" for q in DASS21_QUESTIONS]
    X = df[q_cols].values
    
    y_scores = {
        "dep": df["dep_score"].values,
        "anx": df["anx_score"].values,
        "str": df["str_score"].values
    }
    
    y_classes = {
        "dep": df["dep_class"].values,
        "anx": df["anx_class"].values,
        "str": df["str_class"].values
    }

    print("Step 2: Splitting into 70% Train, 15% Validation, 15% Test...")
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    
    train_scores = {k: v[train_idx] for k, v in y_scores.items()}
    train_classes = {k: v[train_idx] for k, v in y_classes.items()}
    
    test_classes = {k: v[test_idx] for k, v in y_classes.items()}
    test_scores = {k: v[test_idx] for k, v in y_scores.items()}

    print("Step 3: Initializing and Training Multi-Layer Perceptron (ANN)...")
    model = MentalHealthANN()
    model.fit(X_train, train_scores, train_classes)

    print("\n" + "=" * 60)
    print("Step 4: Evaluating ANN Performance on Unseen Test Set (900 samples)")
    print("=" * 60)

    results = {}
    for subscale in ["dep", "anx", "str"]:
        sub_name = "Depression" if subscale == "dep" else ("Anxiety" if subscale == "anx" else "Stress")
        clf = getattr(model, f"clf_{subscale}")
        reg = getattr(model, f"reg_{subscale}")
        
        X_test_scaled = model.scaler.transform(X_test)
        preds_cls = clf.predict(X_test_scaled)
        preds_score = reg.predict(X_test_scaled)
        
        acc = accuracy_score(test_classes[subscale], preds_cls)
        f1 = f1_score(test_classes[subscale], preds_cls, average="weighted")
        mae = np.mean(np.abs(test_scores[subscale] - preds_score))
        
        results[sub_name] = {"Accuracy": acc, "F1-Score": f1, "MAE (Score Points)": mae}
        
        print(f"\n--- {sub_name} Evaluation ---")
        print(f"Classification Accuracy: {acc * 100:.2f}%")
        print(f"Weighted F1-Score:       {f1:.4f}")
        print(f"Regression MAE:          {mae:.2f} score points (Scale: 0-42)")
        print(f"Classification Report:\n{classification_report(test_classes[subscale], preds_cls, zero_division=0)}")

    # Save trained model
    save_path = os.path.join(BACKEND_DIR, "app", "models", "saved", "mental_health_ann.joblib")
    model.save(save_path)
    print(f"\nModel saved successfully to: {save_path}")

    # Generate benchmark summary file
    summary_path = os.path.join(SCRIPT_DIR, "training_summary.json")
    import json
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training summary written to: {summary_path}")

    return model, results

if __name__ == "__main__":
    train_and_evaluate()
