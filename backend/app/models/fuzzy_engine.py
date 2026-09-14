"""
Mamdani Fuzzy Inference System (FIS) for Mental Health Triage & Explainability
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

import numpy as np
from typing import Dict, List, Any, Tuple

class FuzzySet:
    """Represents a fuzzy set with triangular or trapezoidal membership."""
    def __init__(self, name: str, mtype: str, params: List[float]):
        self.name = name
        self.mtype = mtype  # 'trimf' or 'trapmf'
        self.params = params

    def membership(self, x: float | np.ndarray) -> float | np.ndarray:
        if self.mtype == 'trimf':
            a, b, c = self.params
            # Triangular membership function
            with np.errstate(divide='ignore', invalid='ignore'):
                term1 = (x - a) / (b - a) if b != a else np.zeros_like(x)
                term2 = (c - x) / (c - b) if c != b else np.zeros_like(x)
                val = np.maximum(0.0, np.minimum(term1, term2))
            return np.where(np.isnan(val), 0.0, val)
        elif self.mtype == 'trapmf':
            a, b, c, d = self.params
            # Trapezoidal membership function
            with np.errstate(divide='ignore', invalid='ignore'):
                term1 = (x - a) / (b - a) if b != a else np.ones_like(x)
                term2 = (d - x) / (d - c) if d != c else np.ones_like(x)
                val = np.maximum(0.0, np.minimum(np.minimum(term1, 1.0), term2))
            return np.where(np.isnan(val), 0.0, val)
        else:
            raise ValueError(f"Unsupported membership type: {self.mtype}")


class LinguisticVariable:
    """A linguistic variable composed of multiple fuzzy terms."""
    def __init__(self, name: str, universe: Tuple[float, float], terms: Dict[str, FuzzySet]):
        self.name = name
        self.universe = universe
        self.terms = terms

    def get_memberships(self, value: float) -> Dict[str, float]:
        clamped_val = float(np.clip(value, self.universe[0], self.universe[1]))
        return {name: float(fset.membership(clamped_val)) for name, fset in self.terms.items()}


class FuzzyRule:
    """Mamdani Fuzzy Rule with Antecedent and Consequent."""
    def __init__(self, rule_id: int, dep: str, anx: str, str_val: str, consequent: str, description: str):
        self.rule_id = rule_id
        self.dep = dep
        self.anx = anx
        self.str_val = str_val
        self.consequent = consequent
        self.description = description

    def evaluate(self, dep_mu: Dict[str, float], anx_mu: Dict[str, float], str_mu: Dict[str, float]) -> float:
        """Evaluates rule activation weight using Mamdani Min-operator (T-Norm)."""
        w_dep = dep_mu.get(self.dep, 1.0) if self.dep else 1.0
        w_anx = anx_mu.get(self.anx, 1.0) if self.anx else 1.0
        w_str = str_mu.get(self.str_val, 1.0) if self.str_val else 1.0
        return min(w_dep, w_anx, w_str)


class MentalHealthFuzzyEngine:
    """
    Complete Mamdani Fuzzy Inference System for Mental Health Risk Triaging.
    Inputs:
      - Depression (0-42)
      - Anxiety (0-42)
      - Stress (0-42)
    Output:
      - Triage Risk Index (0-100)
    """
    def __init__(self):
        # 1. Linguistic Variables setup
        # Universe 0 to 42 for DASS subscale scores
        self.dep_var = LinguisticVariable(
            "Depression", (0, 42),
            {
                "Normal": FuzzySet("Normal", "trapmf", [0, 0, 7, 10]),
                "Mild": FuzzySet("Mild", "trimf", [8, 11.5, 15]),
                "Moderate": FuzzySet("Moderate", "trimf", [13, 17, 22]),
                "Severe": FuzzySet("Severe", "trimf", [19, 24, 29]),
                "Extremely_Severe": FuzzySet("Extremely_Severe", "trapmf", [26, 30, 42, 42])
            }
        )
        
        self.anx_var = LinguisticVariable(
            "Anxiety", (0, 42),
            {
                "Normal": FuzzySet("Normal", "trapmf", [0, 0, 5, 8]),
                "Mild": FuzzySet("Mild", "trimf", [6, 8.5, 11]),
                "Moderate": FuzzySet("Moderate", "trimf", [9, 12, 16]),
                "Severe": FuzzySet("Severe", "trimf", [14, 17, 21]),
                "Extremely_Severe": FuzzySet("Extremely_Severe", "trapmf", [18, 22, 42, 42])
            }
        )
        
        self.str_var = LinguisticVariable(
            "Stress", (0, 42),
            {
                "Normal": FuzzySet("Normal", "trapmf", [0, 0, 11, 15]),
                "Mild": FuzzySet("Mild", "trimf", [13, 16.5, 20]),
                "Moderate": FuzzySet("Moderate", "trimf", [17, 22, 27]),
                "Severe": FuzzySet("Severe", "trimf", [24, 29.5, 35]),
                "Extremely_Severe": FuzzySet("Extremely_Severe", "trapmf", [32, 36, 42, 42])
            }
        )
        
        # Output: Mental Health Risk Index (0 to 100)
        self.risk_var = LinguisticVariable(
            "Risk_Index", (0, 100),
            {
                "Minimal": FuzzySet("Minimal", "trapmf", [0, 0, 15, 25]),
                "Mild": FuzzySet("Mild", "trimf", [18, 30, 42]),
                "Moderate": FuzzySet("Moderate", "trimf", [35, 52, 68]),
                "Severe": FuzzySet("Severe", "trimf", [60, 75, 88]),
                "Critical": FuzzySet("Critical", "trapmf", [80, 90, 100, 100])
            }
        )

        # 2. Rule Base: 25 Expert Rules modeling clinical triage
        self.rules = self._init_rule_base()
        
        # Discrete resolution for numerical centroid defuzzification
        self.y_resolution = np.linspace(0, 100, 1001)

    def _init_rule_base(self) -> List[FuzzyRule]:
        # Comprehensive clinical triage rule base ensuring complete universe coverage
        # Matrix: Depression (5) x Anxiety (5) with Stress-sensitive modulation
        rules_data = [
            # Normal Depression column
            (1, "Normal", "Normal", None, "Minimal", "Healthy baseline: Depression and Anxiety within normal thresholds."),
            (2, "Normal", "Mild", None, "Mild", "Normal mood with isolated mild anxiety symptoms."),
            (3, "Normal", "Moderate", None, "Moderate", "Significant isolated anxiety symptoms with stable baseline mood."),
            (4, "Normal", "Severe", None, "Severe", "Acute isolated anxiety/panic disorder symptoms requiring clinical attention."),
            (5, "Normal", "Extremely_Severe", None, "Critical", "Extreme isolated panic/anxiety crisis requiring immediate evaluation."),

            # Mild Depression column
            (6, "Mild", "Normal", None, "Mild", "Mild isolated dysphoria; adaptive lifestyle coping recommended."),
            (7, "Mild", "Mild", None, "Mild", "Concurrent low-grade dysphoria and nervousness; self-monitoring advised."),
            (8, "Mild", "Moderate", None, "Moderate", "Moderate anxiety coupled with mild depressive affect."),
            (9, "Mild", "Severe", None, "Severe", "Severe autonomic anxiety with secondary depressive symptoms."),
            (10, "Mild", "Extremely_Severe", None, "Critical", "Acute panic vulnerability alongside mild depressive baseline."),

            # Moderate Depression column
            (11, "Moderate", "Normal", None, "Moderate", "Moderate clinical depression with low anxiety; counseling recommended."),
            (12, "Moderate", "Mild", None, "Moderate", "Moderate depressive dysphoria with mild anxiety agitation."),
            (13, "Moderate", "Moderate", None, "Moderate", "Dual moderate depression and anxiety syndrome."),
            (14, "Moderate", "Severe", None, "Severe", "Severe anxiety with concurrent moderate depressive disorder."),
            (15, "Moderate", "Extremely_Severe", None, "Critical", "Extreme acute anxiety superimposed on moderate depression."),

            # Severe Depression column
            (16, "Severe", "Normal", None, "Severe", "Severe clinical depression with vegetative symptoms; psychiatric intake advised."),
            (17, "Severe", "Mild", None, "Severe", "Severe depressive state with mild accompanying anxiety."),
            (18, "Severe", "Moderate", None, "Severe", "Severe depression compounded by moderate clinical anxiety."),
            (19, "Severe", "Severe", None, "Severe", "Dual severe depressive and anxiety disorder requiring urgent intervention."),
            (20, "Severe", "Extremely_Severe", None, "Critical", "Profound acute distress: severe depression plus extreme anxiety."),

            # Extremely Severe Depression column
            (21, "Extremely_Severe", "Normal", None, "Critical", "Extreme depressive crisis requiring immediate psychiatric safety evaluation."),
            (22, "Extremely_Severe", "Mild", None, "Critical", "Critical depressive episode with mild anxiety features."),
            (23, "Extremely_Severe", "Moderate", None, "Critical", "Critical depressive symptoms accompanied by moderate tension."),
            (24, "Extremely_Severe", "Severe", None, "Critical", "Emergency psychiatric priority: critical depression and severe anxiety."),
            (25, "Extremely_Severe", "Extremely_Severe", None, "Critical", "Maximum critical priority: acute severity across primary affective dimensions."),

            # Stress Modulation & Escalation Rules
            (26, None, None, "Extremely_Severe", "Critical", "Acute incapacitating stress overload indicating burnout crisis."),
            (27, "Moderate", None, "Severe", "Severe", "Moderate affective vulnerability exacerbated by high environmental stress."),
            (28, None, "Moderate", "Severe", "Severe", "Moderate anxiety heightened to high vulnerability under intense stress."),
            (29, "Mild", "Mild", "Moderate", "Moderate", "Low-grade mood and anxiety symptoms aggravated into moderate distress by stress."),
            (30, "Normal", "Normal", "Extremely_Severe", "Severe", "High acute situational stress threatening psychological decompensation.")
        ]

        rules = []
        for rid, d, a, s, cons, desc in rules_data:
            rules.append(FuzzyRule(rid, d, a, s, cons, desc))
        return rules

    def infer(self, dep_score: float, anx_score: float, str_score: float) -> Dict[str, Any]:
        """
        Executes Mamdani Fuzzy Inference Pipeline:
        1. Fuzzification
        2. Rule Activation (Min-Operator)
        3. Aggregation (Max-Operator)
        4. Defuzzification (Centroid)
        """
        # Step 1: Fuzzification
        dep_mu = self.dep_var.get_memberships(dep_score)
        anx_mu = self.anx_var.get_memberships(anx_score)
        str_mu = self.str_var.get_memberships(str_score)

        # Step 2: Rule Evaluation
        fired_rules = []
        consequent_activations = {
            "Minimal": 0.0,
            "Mild": 0.0,
            "Moderate": 0.0,
            "Severe": 0.0,
            "Critical": 0.0
        }

        for rule in self.rules:
            weight = rule.evaluate(dep_mu, anx_mu, str_mu)
            if weight > 0.001:
                fired_rules.append({
                    "rule_id": rule.rule_id,
                    "description": rule.description,
                    "consequent": rule.consequent,
                    "weight": round(float(weight), 4)
                })
                consequent_activations[rule.consequent] = max(
                    consequent_activations[rule.consequent], float(weight)
                )

        # Sort fired rules by activation weight
        fired_rules.sort(key=lambda r: r["weight"], reverse=True)

        # Step 3: Aggregation over the output universe
        aggregated_mu = np.zeros_like(self.y_resolution)
        for cname, alpha in consequent_activations.items():
            if alpha > 0.0:
                fset = self.risk_var.terms[cname]
                # Mamdani clipping (min with alpha)
                clipped = np.minimum(alpha, fset.membership(self.y_resolution))
                # Maximum aggregation
                aggregated_mu = np.maximum(aggregated_mu, clipped)

        # Step 4: Centroid Defuzzification
        area = np.sum(aggregated_mu)
        if area > 1e-6:
            risk_index = float(np.sum(self.y_resolution * aggregated_mu) / area)
        else:
            # Fallback based on weighted inputs
            risk_index = float(np.clip((dep_score + anx_score + 0.5 * str_score) / (42 + 42 + 21) * 100, 0, 100))

        # Determine triage classification from risk index
        triage_category, triage_color = self._get_triage_tier(risk_index)
        action_plan = self._get_action_plan(triage_category)

        return {
            "risk_index": round(risk_index, 1),
            "triage_category": triage_category,
            "triage_color": triage_color,
            "fuzzified_inputs": {
                "Depression": {k: round(v, 3) for k, v in dep_mu.items() if v > 0.01},
                "Anxiety": {k: round(v, 3) for k, v in anx_mu.items() if v > 0.01},
                "Stress": {k: round(v, 3) for k, v in str_mu.items() if v > 0.01},
            },
            "consequent_activations": {k: round(v, 3) for k, v in consequent_activations.items()},
            "fired_rules_count": len(fired_rules),
            "top_fired_rules": fired_rules[:5],
            "action_plan": action_plan
        }

    def _get_triage_tier(self, risk_index: float) -> Tuple[str, str]:
        if risk_index < 25.0:
            return "Minimal Risk", "#10b981"    # emerald green
        elif risk_index < 45.0:
            return "Mild Vulnerability", "#06b6d4"  # cyan/blue
        elif risk_index < 68.0:
            return "Moderate Concern", "#f59e0b"   # amber
        elif risk_index < 85.0:
            return "High Risk / Severe", "#f97316" # orange/red
        else:
            return "Critical Priority", "#ef4444"  # bright red

    def _get_action_plan(self, tier: str) -> Dict[str, Any]:
        plans = {
            "Minimal Risk": {
                "title": "Healthy State & Preventative Care",
                "recommendation": "Your indicators are within healthy adaptive limits. Maintain healthy sleep hygiene, regular physical activity, and social connections.",
                "urgency": "Low / Routine",
                "next_steps": ["Maintain regular exercise", "Practice daily mindfulness", "Retest periodically"]
            },
            "Mild Vulnerability": {
                "title": "Early Stress Management & Self-Care",
                "recommendation": "Mild affective or tension fluctuations observed. Structured stress reduction and talking to close mentors or counselors is advised.",
                "urgency": "Moderate / Self-Monitoring",
                "next_steps": ["Deep breathing & relaxation routines", "Limit late-night screen time", "Utilize university wellness resources"]
            },
            "Moderate Concern": {
                "title": "Clinical Consultation Recommended",
                "recommendation": "Symptoms are causing non-trivial psychological strain. We recommend consulting a licensed psychologist or student counseling services.",
                "urgency": "High / Schedule Assessment",
                "next_steps": ["Schedule a session with university counseling", "Engage in cognitive reframing", "Involve a trusted support person"]
            },
            "High Risk / Severe": {
                "title": "Formal Psychiatric & Clinical Support Needed",
                "recommendation": "Prominent clinical anxiety and depressive symptoms detected. Immediate scheduling with a qualified mental health clinician or psychiatrist is strongly recommended.",
                "urgency": "Urgent / Prompt Clinical Attention",
                "next_steps": ["Book an urgent clinical intake appointment", "Refrain from solitary isolation", "Notify academic advisors if academic leave is needed"]
            },
            "Critical Priority": {
                "title": "Immediate Crisis Intervention",
                "recommendation": "Severe acute psychological distress detected. Please seek immediate support from professional psychiatric crisis hotlines or emergency health services.",
                "urgency": "Immediate / Emergency Priority",
                "next_steps": ["Contact National Tele-MANAS (14416) or local helpline", "Reach out to campus medical center immediately", "Stay accompanied by trusted family or friends"]
            }
        }
        return plans.get(tier, plans["Moderate Concern"])


if __name__ == "__main__":
    engine = MentalHealthFuzzyEngine()
    result = engine.infer(dep_score=18.0, anx_score=14.0, str_score=22.0)
    print("Fuzzy Inference Test Result:")
    print(f"Risk Index: {result['risk_index']}%")
    print(f"Triage Category: {result['triage_category']}")
    print(f"Top Fired Rule: {result['top_fired_rules'][0] if result['top_fired_rules'] else 'None'}")
