"""
DASS-21 Dataset Loader and Clinical Rubric Definition
Author: Soft Computing Project Team
Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Standard DASS-21 Question Inventory
DASS21_QUESTIONS = [
    {"id": 1, "subscale": "Stress", "text": "I found it hard to wind down"},
    {"id": 2, "subscale": "Anxiety", "text": "I was aware of dryness of my mouth"},
    {"id": 3, "subscale": "Depression", "text": "I couldn't seem to experience any positive feeling at all"},
    {"id": 4, "subscale": "Anxiety", "text": "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)"},
    {"id": 5, "subscale": "Depression", "text": "I found it difficult to work up the initiative to do things"},
    {"id": 6, "subscale": "Stress", "text": "I tended to over-react to situations"},
    {"id": 7, "subscale": "Anxiety", "text": "I experienced trembling (e.g. in the hands)"},
    {"id": 8, "subscale": "Stress", "text": "I felt that I was using a lot of nervous energy"},
    {"id": 9, "subscale": "Anxiety", "text": "I was worried about situations in which I might panic and make a fool of myself"},
    {"id": 10, "subscale": "Depression", "text": "I felt that I had nothing to look forward to"},
    {"id": 11, "subscale": "Stress", "text": "I found myself getting agitated"},
    {"id": 12, "subscale": "Stress", "text": "I found it difficult to relax"},
    {"id": 13, "subscale": "Depression", "text": "I felt down-hearted and blue"},
    {"id": 14, "subscale": "Stress", "text": "I was intolerant of anything that kept me from getting on with what I was doing"},
    {"id": 15, "subscale": "Anxiety", "text": "I felt I was close to panic"},
    {"id": 16, "subscale": "Depression", "text": "I was unable to become enthusiastic about anything"},
    {"id": 17, "subscale": "Depression", "text": "I felt I wasn't worth much as a person"},
    {"id": 18, "subscale": "Stress", "text": "I felt that I was rather touchy"},
    {"id": 19, "subscale": "Anxiety", "text": "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)"},
    {"id": 20, "subscale": "Anxiety", "text": "I felt scared without any good reason"},
    {"id": 21, "subscale": "Depression", "text": "I felt that life was meaningless"}
]

# Clinical cutoff ranges (DASS-21 raw score * 2)
SEVERITY_LEVELS = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

CUTOFFS = {
    "Depression": [
        (0, 9, "Normal", 0),
        (10, 13, "Mild", 1),
        (14, 20, "Moderate", 2),
        (21, 27, "Severe", 3),
        (28, 42, "Extremely Severe", 4),
    ],
    "Anxiety": [
        (0, 7, "Normal", 0),
        (8, 9, "Mild", 1),
        (10, 14, "Moderate", 2),
        (15, 19, "Severe", 3),
        (20, 42, "Extremely Severe", 4),
    ],
    "Stress": [
        (0, 14, "Normal", 0),
        (15, 18, "Mild", 1),
        (19, 25, "Moderate", 2),
        (26, 33, "Severe", 3),
        (34, 42, "Extremely Severe", 4),
    ]
}

DEPRESSION_INDICES = [i for i, q in enumerate(DASS21_QUESTIONS) if q["subscale"] == "Depression"]
ANXIETY_INDICES = [i for i, q in enumerate(DASS21_QUESTIONS) if q["subscale"] == "Anxiety"]
STRESS_INDICES = [i for i, q in enumerate(DASS21_QUESTIONS) if q["subscale"] == "Stress"]

def score_to_category(score: float, subscale: str) -> Tuple[str, int]:
    """Maps multiplied score (0-42) to clinical category name and class id (0-4)."""
    for low, high, label, cid in CUTOFFS[subscale]:
        if low <= score <= high:
            return label, cid
    return "Extremely Severe", 4

def generate_clinically_calibrated_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic DASS-21 dataset mirroring empirical psychometric studies:
    - Multidimensional latent traits (Depression, Anxiety, Stress) with realistic covariance (~0.6-0.7)
    - Realistic Likert scale discretization [0, 3] via cumulative threshold mapping
    - Ground truth continuous scores and clinical discrete categories
    """
    np.random.seed(random_state)
    
    # Latent mean and covariance matrix modeling clinical co-morbidity
    mean = [0.0, 0.0, 0.0]
    cov = [
        [1.0, 0.65, 0.60],  # Depression
        [0.65, 1.0, 0.70],  # Anxiety
        [0.60, 0.70, 1.0]   # Stress
    ]
    
    latent = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    # Generate question responses for 21 items
    data = {}
    for i, q in enumerate(DASS21_QUESTIONS):
        sub = q["subscale"]
        latent_idx = 0 if sub == "Depression" else (1 if sub == "Anxiety" else 2)
        
        # Add item-specific difficulty and noise
        item_signal = latent[:, latent_idx] + np.random.normal(0, 0.55, size=n_samples)
        
        # Discretize into 0, 1, 2, 3 using realistic thresholds
        thresholds = [-0.4, 0.4, 1.2]
        responses = np.zeros(n_samples, dtype=int)
        responses[item_signal > thresholds[0]] = 1
        responses[item_signal > thresholds[1]] = 2
        responses[item_signal > thresholds[2]] = 3
        
        data[f"Q{q['id']}"] = responses

    df = pd.DataFrame(data)
    
    # Calculate raw subscale sums and multiplied scores
    dep_cols = [f"Q{q['id']}" for q in DASS21_QUESTIONS if q["subscale"] == "Depression"]
    anx_cols = [f"Q{q['id']}" for q in DASS21_QUESTIONS if q["subscale"] == "Anxiety"]
    str_cols = [f"Q{q['id']}" for q in DASS21_QUESTIONS if q["subscale"] == "Stress"]
    
    df["dep_raw"] = df[dep_cols].sum(axis=1)
    df["anx_raw"] = df[anx_cols].sum(axis=1)
    df["str_raw"] = df[str_cols].sum(axis=1)
    
    # Standard DASS-21 to DASS-42 multiplied scale (* 2)
    df["dep_score"] = df["dep_raw"] * 2
    df["anx_score"] = df["anx_raw"] * 2
    df["str_score"] = df["str_raw"] * 2
    
    # Categorical and numerical class targets
    df["dep_class"] = df["dep_score"].apply(lambda s: score_to_category(s, "Depression")[1])
    df["anx_class"] = df["anx_score"].apply(lambda s: score_to_category(s, "Anxiety")[1])
    df["str_class"] = df["str_score"].apply(lambda s: score_to_category(s, "Stress")[1])
    
    df["dep_label"] = df["dep_score"].apply(lambda s: score_to_category(s, "Depression")[0])
    df["anx_label"] = df["anx_score"].apply(lambda s: score_to_category(s, "Anxiety")[0])
    df["str_label"] = df["str_score"].apply(lambda s: score_to_category(s, "Stress")[0])
    
    return df

if __name__ == "__main__":
    df = generate_clinically_calibrated_dataset(500)
    print("Dataset generated successfully:")
    print(df.head())
    print("Depression distribution:\n", df["dep_label"].value_counts())
    print("Anxiety distribution:\n", df["anx_label"].value_counts())
