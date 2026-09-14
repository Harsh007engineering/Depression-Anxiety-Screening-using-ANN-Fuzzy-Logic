"""
Artificial Neural Network (ANN) Module for DASS-21 Severity Estimation
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

class MentalHealthANN:
    """
    Multi-Layer Perceptron (MLP) Neural Network for Mental Health Screening.
    Takes 21 DASS item responses and predicts:
      1. Continuous symptom severity scores for Depression, Anxiety, and Stress [0, 42]
      2. Multi-class severity classification [0: Normal, 1: Mild, 2: Moderate, 3: Severe, 4: Extremely Severe]
    """
    def __init__(self):
        # 1. Regressors to predict calibrated continuous latent scores
        # Architecture: 21 -> 64 -> 32 -> 1
        self.reg_dep = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=600, random_state=42, alpha=0.01)
        self.reg_anx = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=600, random_state=42, alpha=0.01)
        self.reg_str = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=600, random_state=42, alpha=0.01)

        # 2. Classifiers for discrete diagnostic staging
        # Architecture: 21 -> 64 -> 32 -> 5 classes
        self.clf_dep = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=600, random_state=42, early_stopping=True)
        self.clf_anx = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=600, random_state=42, early_stopping=True)
        self.clf_str = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=600, random_state=42, early_stopping=True)

        self.scaler = StandardScaler()
        self.is_trained = False

    def fit(self, X: np.ndarray, y_scores: Dict[str, np.ndarray], y_classes: Dict[str, np.ndarray]):
        """Trains the MLP networks on survey features."""
        X_scaled = self.scaler.fit_transform(X)

        # Train continuous regressors
        self.reg_dep.fit(X_scaled, y_scores["dep"])
        self.reg_anx.fit(X_scaled, y_scores["anx"])
        self.reg_str.fit(X_scaled, y_scores["str"])

        # Train discrete classifiers
        self.clf_dep.fit(X_scaled, y_classes["dep"])
        self.clf_anx.fit(X_scaled, y_classes["anx"])
        self.clf_str.fit(X_scaled, y_classes["str"])

        self.is_trained = True

    def predict(self, responses_21: list | np.ndarray) -> Dict[str, Any]:
        """
        Runs ANN forward pass on a single user's 21-item questionnaire.
        Returns predicted scores, predicted discrete classes, and probability distributions.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call fit() or load() before predict().")

        x = np.array(responses_21, dtype=float).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        # Continuous predictions clipped to DASS range [0, 42]
        pred_dep_score = float(np.clip(self.reg_dep.predict(x_scaled)[0], 0.0, 42.0))
        pred_anx_score = float(np.clip(self.reg_anx.predict(x_scaled)[0], 0.0, 42.0))
        pred_str_score = float(np.clip(self.reg_str.predict(x_scaled)[0], 0.0, 42.0))

        # Discrete classes and probability distributions
        dep_class = int(self.clf_dep.predict(x_scaled)[0])
        anx_class = int(self.clf_anx.predict(x_scaled)[0])
        str_class = int(self.clf_str.predict(x_scaled)[0])

        dep_probs = [float(p) for p in self.clf_dep.predict_proba(x_scaled)[0]]
        anx_probs = [float(p) for p in self.clf_anx.predict_proba(x_scaled)[0]]
        str_probs = [float(p) for p in self.clf_str.predict_proba(x_scaled)[0]]

        severity_labels = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

        return {
            "depression": {
                "score": round(pred_dep_score, 1),
                "class_id": dep_class,
                "label": severity_labels[dep_class] if dep_class < len(severity_labels) else "Extremely Severe",
                "probabilities": {severity_labels[i]: round(dep_probs[i], 3) for i in range(len(dep_probs))}
            },
            "anxiety": {
                "score": round(pred_anx_score, 1),
                "class_id": anx_class,
                "label": severity_labels[anx_class] if anx_class < len(severity_labels) else "Extremely Severe",
                "probabilities": {severity_labels[i]: round(anx_probs[i], 3) for i in range(len(anx_probs))}
            },
            "stress": {
                "score": round(pred_str_score, 1),
                "class_id": str_class,
                "label": severity_labels[str_class] if str_class < len(severity_labels) else "Extremely Severe",
                "probabilities": {severity_labels[i]: round(str_probs[i], 3) for i in range(len(str_probs))}
            }
        }

    def save(self, filepath: str):
        """Saves model weights and scaler to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "reg_dep": self.reg_dep,
            "reg_anx": self.reg_anx,
            "reg_str": self.reg_str,
            "clf_dep": self.clf_dep,
            "clf_anx": self.clf_anx,
            "clf_str": self.clf_str,
            "is_trained": self.is_trained
        }, filepath)

    def load(self, filepath: str):
        """Loads model weights and scaler from disk."""
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.reg_dep = data["reg_dep"]
        self.reg_anx = data["reg_anx"]
        self.reg_str = data["reg_str"]
        self.clf_dep = data["clf_dep"]
        self.clf_anx = data["clf_anx"]
        self.clf_str = data["clf_str"]
        self.is_trained = data["is_trained"]
