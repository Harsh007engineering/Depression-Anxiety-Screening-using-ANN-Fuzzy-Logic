"""
Hybrid Neuro-Fuzzy Integration Pipeline
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

import os
from typing import List, Dict, Any
from .ann_model import MentalHealthANN
from .fuzzy_engine import MentalHealthFuzzyEngine

class MentalHealthHybridPipeline:
    """
    Two-Stage Hybrid Neuro-Fuzzy Screening Architecture:
      Stage 1: ANN Multi-Layer Perceptron (Non-linear latent symptom scoring & pattern weighting)
      Stage 2: Mamdani Fuzzy Inference System (Linguistic triage, ambiguity resolution & explainability)
    """
    def __init__(self, ann_weights_path: str = None):
        self.ann = MentalHealthANN()
        self.fuzzy = MentalHealthFuzzyEngine()
        self.weights_path = ann_weights_path

        if ann_weights_path and os.path.exists(ann_weights_path):
            self.ann.load(ann_weights_path)

    def screen(self, responses_21: List[int], contextual_stress_factor: float = 0.0) -> Dict[str, Any]:
        """
        Executes end-to-end neuro-fuzzy screening for a 21-item questionnaire.
        """
        if len(responses_21) != 21:
            raise ValueError(f"Expected 21 questionnaire items, got {len(responses_21)}")

        # 1. Classical Raw Rubric (DASS-21 Subscale Sums * 2)
        # Depression: Q3, Q5, Q10, Q13, Q16, Q17, Q21 (0-indexed: 2, 4, 9, 12, 15, 16, 20)
        dep_indices = [2, 4, 9, 12, 15, 16, 20]
        anx_indices = [1, 3, 6, 8, 14, 18, 19]
        str_indices = [0, 5, 7, 10, 11, 13, 17]

        raw_dep = sum(responses_21[i] for i in dep_indices) * 2
        raw_anx = sum(responses_21[i] for i in anx_indices) * 2
        raw_str = sum(responses_21[i] for i in str_indices) * 2

        # 2. Stage 1: ANN Forward Pass
        if self.ann.is_trained:
            ann_preds = self.ann.predict(responses_21)
            eval_dep_score = ann_preds["depression"]["score"]
            eval_anx_score = ann_preds["anx"]["score"] if "anx" in ann_preds else ann_preds["anxiety"]["score"]
            eval_str_score = ann_preds["stress"]["score"]
        else:
            # Baseline deterministic mapping if ANN weights haven't been loaded yet
            eval_dep_score = float(raw_dep)
            eval_anx_score = float(raw_anx)
            eval_str_score = float(raw_str)
            ann_preds = {
                "depression": {"score": raw_dep, "label": "Estimated", "probabilities": {}},
                "anxiety": {"score": raw_anx, "label": "Estimated", "probabilities": {}},
                "stress": {"score": raw_str, "label": "Estimated", "probabilities": {}}
            }

        # Optional modulation by contextual stressor
        if contextual_stress_factor > 0:
            eval_str_score = min(42.0, eval_str_score + (contextual_stress_factor * 2.0))

        # 3. Stage 2: Mamdani Fuzzy Inference Engine
        fuzzy_results = self.fuzzy.infer(
            dep_score=eval_dep_score,
            anx_score=eval_anx_score,
            str_score=eval_str_score
        )

        # 4. Synthesize Final Screening Report
        return {
            "dass21_raw_scores": {
                "depression": raw_dep,
                "anxiety": raw_anx,
                "stress": raw_str
            },
            "ann_projections": ann_preds,
            "fuzzy_triage": fuzzy_results,
            "summary": {
                "risk_index": fuzzy_results["risk_index"],
                "triage_category": fuzzy_results["triage_category"],
                "triage_color": fuzzy_results["triage_color"],
                "dominant_concern": self._determine_dominant_concern(eval_dep_score, eval_anx_score, eval_str_score)
            }
        }

    def _determine_dominant_concern(self, dep: float, anx: float, stress: float) -> str:
        # Normalized by max possible (42)
        ratios = {"Depression": dep / 42.0, "Anxiety": anx / 42.0, "Stress": stress / 42.0}
        max_sub = max(ratios, key=ratios.get)
        if ratios[max_sub] < 0.2:
            return "None (Healthy Baseline)"
        return max_sub
