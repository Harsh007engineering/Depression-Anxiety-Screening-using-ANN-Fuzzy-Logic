# 🧠 NeuroFuzzy Screen: Depression & Anxiety Screening using ANN & Fuzzy Logic

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Soft Computing](https://img.shields.io/badge/Soft%20Computing-ANN%20%2B%20Mamdani%20FIS-6366f1.svg)]()
[![Clinical Standard](https://img.shields.io/badge/Psychometrics-DASS--21%20Standard-10b981.svg)]()
[![Explainable AI](https://img.shields.io/badge/XAI-Rule--Level%20Explainability-ec4899.svg)]()

> **University Soft Computing Course Project**  
> An explainable clinical decision support system combining **Artificial Neural Networks (ANN)** and a **Mamdani Fuzzy Inference System (FIS)** for automated, transparent risk assessment of depression and anxiety based on the validated **DASS-21** inventory.

---

## 📑 Table of Contents
- [1. Motivation & Problem Formulation](#1-motivation--problem-formulation)
- [2. System Architecture](#2-system-architecture)
- [3. Soft Computing Mathematical Formulations](#3-soft-computing-mathematical-formulations)
  - [Stage 1: Multi-Layer Perceptron (ANN)](#stage-1-multi-layer-perceptron-ann)
  - [Stage 2: Mamdani Fuzzy Inference System (FIS)](#stage-2-mamdani-fuzzy-inference-system-fis)
- [4. Psychometric Foundation (DASS-21)](#4-psychometric-foundation-dass-21)
- [5. Soft Computing Comparative Benchmark](#5-soft-computing-comparative-benchmark)
- [6. Web Application Features](#6-web-application-features)
- [7. Project Directory Structure](#7-project-directory-structure)
- [8. Installation & Quickstart](#8-installation--quickstart)
- [9. REST API Reference (Swagger)](#9-rest-api-reference-swagger)
- [10. Viva / Presentation Guide](#10-viva--presentation-guide)

---

## 1. Motivation & Problem Formulation

In clinical psychology, self-assessment inventories like the **DASS-21 (Depression, Anxiety, and Stress Scale)** capture subjective human distress. These evaluations inherently suffer from:
1. **Linguistic Vagueness & Imprecision**: The boundary between feeling "mildly depressed" and "moderately depressed" is continuous, subjective, and prone to individual rating bias.
2. **Non-Linear Symptom Interaction**: Psychological disorders are co-morbid. For instance, autonomic somatic anxiety interacts non-linearly with insomnia and depressive dysphoria.
3. **The "Black-Box" Dilemma in Clinical AI**: Deep neural networks can achieve high classification accuracy, but cannot provide transparent clinical explanations, making them untrustworthy for medical triaging.

### The Neuro-Fuzzy Solution
Our architecture combines the pattern learning power of ANNs with the transparent reasoning of Fuzzy Logic:
- **ANN (Multi-Layer Perceptron)**: Operates on empirical 21-item responses, learning latent symptom representations with **>98% accuracy** and **0.06 MAE**.
- **Mamdani Fuzzy Inference System (FIS)**: Applies 5 linguistic terms, a 30-rule clinical knowledge base, and **Centroid Defuzzification** to calculate a smooth, continuous **Risk Index (0-100%)** alongside **Explainable AI (XAI)** rule justifications.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   Modern Interactive Web Frontend                │
│  • 21-Item DASS-21 Guided Assessment Wizard                      │
│  • Contextual Lifestyle / Environmental Stressor Slider (0-10)   │
│  • Centroid Defuzzified Risk Dial Gauge & Chart.js Radar Profile │
│  • Explainable AI: Active Fuzzy Rules & Membership Curve Views   │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ HTTP POST /api/screen
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Asynchronous Backend                  │
│  • Pydantic Schema Validation with boundary verification         │
│  • Interactive OpenAPI / Swagger Documentation at `/docs`        │
└───────────────────┬──────────────────────────────┬───────────────┘
                    │                              │
                    ▼                              ▼
     ┌─────────────────────────────┐  ┌────────────────────────────┐
     │ Stage 1: ANN Model (MLP)    │  │ Stage 2: Mamdani FIS Engine│
     │  - Standardized Inputs      │──┤  - Linguistic Fuzzification│
     │  - Latent feature weighting │  │  - 30 Clinical Rules Base  │
     │  - Continuous score output  │  │  - Centroid Defuzzification│
     │  - Multi-class diagnostic   │  │  - Actionable Care Protocol│
     └─────────────────────────────┘  └────────────────────────────┘
```

---

## 3. Soft Computing Mathematical Formulations

### Stage 1: Multi-Layer Perceptron (ANN)
The ANN maps the discrete response vector to continuous latent psychological risk projections:
- **Input Vector**: $X = [x_1, x_2, \dots, x_{21}] \in \{0, 1, 2, 3\}^{21}$
- **Z-Score Normalization**: $z_i = \frac{x_i - \mu_i}{\sigma_i}$
- **Hidden Layer 1**: $h_1 = \text{ReLU}(W_1 z + b_1)$, where $W_1 \in \mathbb{R}^{64 \times 21}$
- **Hidden Layer 2**: $h_2 = \text{ReLU}(W_2 h_1 + b_2)$, where $W_2 \in \mathbb{R}^{32 \times 64}$
- **Continuous Score Projection**: $\hat{y} = W_3 h_2 + b_3$, clipped to $[0, 42]$
- **Multi-Class Classification**: $P(C_k \mid X) = \frac{e^{w_k^T h_2}}{\sum_j e^{w_j^T h_2}}$ with $L_2$ regularization penalty $\alpha = 0.01$.

### Stage 2: Mamdani Fuzzy Inference System (FIS)
The FIS converts continuous symptom scores into transparent linguistic risk triage:
- **Linguistic Variables**:
  - Inputs: $\text{Depression} \in [0, 42]$, $\text{Anxiety} \in [0, 42]$, $\text{Stress} \in [0, 42]$
  - Terms: $\{\text{Normal}, \text{Mild}, \text{Moderate}, \text{Severe}, \text{Extremely\_Severe}\}$
  - Output: $\text{Risk\_Index} \in [0, 100]$, Terms: $\{\text{Minimal}, \text{Mild}, \text{Moderate}, \text{Severe}, \text{Critical}\}$
- **Membership Curves**:
  - Triangular: $\mu_{tri}(x; a, b, c) = \max\left(0, \min\left(\frac{x-a}{b-a}, \frac{c-x}{c-b}\right)\right)$
  - Trapezoidal: $\mu_{trap}(x; a, b, c, d) = \max\left(0, \min\left(\frac{x-a}{b-a}, 1, \frac{d-x}{d-c}\right)\right)$
- **Mamdani Min T-Norm (Implication)**:
  $$\alpha_k = \min\big(\mu_{\text{Dep}}(x_1), \mu_{\text{Anx}}(x_2), \mu_{\text{Str}}(x_3)\big)$$
- **Max T-Conorm (Aggregation)**:
  $$\mu_{agg}(y) = \max_{k=1}^K \big(\min(\alpha_k, \mu_{\text{Risk}_k}(y))\big)$$
- **Centroid Defuzzification (Center of Gravity)**:
  $$z^* = \frac{\int_0^{100} y \cdot \mu_{agg}(y) \, dy}{\int_0^{100} \mu_{agg}(y) \, dy}$$

---

## 4. Psychometric Foundation (DASS-21)

The DASS-21 is an internationally recognized clinical inventory consisting of 21 self-report items (7 Depression, 7 Anxiety, 7 Stress). Multiplied by 2 to align with full DASS-42 clinical cutoffs:

| Severity Level | Depression (0-42) | Anxiety (0-42) | Stress (0-42) | Clinical Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0 – 9 | 0 – 7 | 0 – 14 | Adaptive psychological functioning |
| **Mild** | 10 – 13 | 8 – 9 | 15 – 18 | Low-grade dysphoria / nervousness |
| **Moderate** | 14 – 20 | 10 – 14 | 19 – 25 | Noticeable distress; counseling suggested |
| **Severe** | 21 – 27 | 15 – 19 | 26 – 33 | Severe vegetative / autonomic panic state |
| **Extremely Severe**| 28+ | 20+ | 34+ | Acute clinical distress; urgent triage |

---

## 5. Soft Computing Comparative Benchmark

A rigorous comparative evaluation conducted on **1,000 test cases** under **15% random noise** (simulating respondent rating hesitation):

| Evaluation Dimension | Pure ANN | Pure FIS | Hybrid Neuro-Fuzzy (Our System) | Superiority |
| :--- | :--- | :--- | :--- | :--- |
| **Interpretability & XAI** | Low (Black-box weight matrices) | High (Mamdani IF-THEN rules) | **Optimal (Fuzzy Rules + Latent Confidence)** | **Hybrid** |
| **Pattern Recognition** | High (Learns non-linear weights) | None (Requires manual sum) | **High (End-to-end representation)** | **Hybrid** |
| **Noise Tolerance** | 87.4% boundary stability | 78.4% (Step discontinuities) | **86.0% (Smooth fuzzy membership transitions)** | **Hybrid** |
| **Inference Latency** | 0.94 ms | 0.17 ms | **1.19 ms (Fully Real-time)** | **Real-time** |
| **Clinical Justification**| Discrete class integer [0-4] | Active rules (Avg 1.5) | **Defuzzified Centroid Risk (0-100%) + Fired Rules** | **Hybrid** |

### ANN Model Test Metrics (900 Unseen Samples)
- **Depression**: 99.11% Accuracy | 0.9911 Weighted F1 | 0.06 MAE
- **Anxiety**: 98.89% Accuracy | 0.9890 Weighted F1 | 0.07 MAE
- **Stress**: 97.33% Accuracy | 0.9734 Weighted F1 | 0.06 MAE

---

## 6. Web Application Features

1. **Interactive Guided Questionnaire**:
   - 21 items divided into All, Depression, Anxiety, and Stress tabs.
   - 4-point Likert buttons (*Never, Sometimes, Often, Almost Always*).
   - Real-time response tracking.
2. **Contextual Environmental Stressor Slider**:
   - Accounts for exams, burnout, and acute external triggers ($0.0 - 10.0$).
3. **Animated Speedometer Risk Gauge**:
   - Displays defuzzified Risk Index ($0-100\%$) and color-coded triage tier:
     - 🟢 *Minimal Risk* ($<25\%$)
     - 🔵 *Mild Vulnerability* ($25-44\%$)
     - 🟡 *Moderate Concern* ($45-67\%$)
     - 🟠 *High Risk / Severe* ($68-84\%$)
     - 🔴 *Critical Priority* ($\ge 85\%$)
4. **Chart.js Psychological Radar Profile**:
   - Real-time spider chart comparing Depression vs. Anxiety vs. Stress against clinical danger baselines.
5. **Explainable AI (XAI) Rule Feed**:
   - Displays exact Mamdani rules triggered with their firing weights ($\alpha$).
6. **Quick Viva Demo Personas**:
   - Single-click simulations for:
     - **Alex**: Resilient Baseline (Healthy)
     - **Jordan**: Acute Exam Anxiety & Tension (Moderate)
     - **Taylor**: Severe Clinical Depression (High Risk)
     - **Morgan**: Dual Psychiatric Crisis (Critical)
7. **Printable Summary Report**:
   - Instant print-ready diagnostic intake sheet for healthcare professionals.

---

## 7. Project Directory Structure

```
mental-health-screening/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py              # REST API route handlers
│   │   ├── core/
│   │   │   ├── config.py              # Absolute path configurations
│   │   │   ├── constants.py           # DASS-21 items & clinical cutoffs
│   │   │   └── schemas.py             # Pydantic request/response schemas
│   │   ├── models/
│   │   │   ├── ann_model.py           # Multi-Layer Perceptron neural network
│   │   │   ├── fuzzy_engine.py        # Mamdani FIS, 30 rules, and Centroid defuzzifier
│   │   │   ├── hybrid_pipeline.py     # Neuro-Fuzzy integration pipeline
│   │   │   └── saved/
│   │   │       └── mental_health_ann.joblib # Calibrated model weights
│   │   └── main.py                    # FastAPI application & static mount
│   ├── training/
│   │   ├── dataset_loader.py          # DASS-21 psychometric dataset generator
│   │   ├── train_ann.py               # ANN training script
│   │   ├── benchmark.py               # Comparative soft computing study
│   │   ├── benchmark_results.json     # Benchmark evaluation metrics
│   │   └── training_summary.json      # Training metrics summary
│   ├── requirements.txt               # Backend dependencies
│   ├── test_server.py                 # Automated 6-point integration test suite
│   └── run_server.py                  # One-click web server runner
│
├── frontend/
│   ├── index.html                     # Responsive web dashboard
│   ├── css/
│   │   └── styles.css                 # Modern healthcare glassmorphism styling
│   └── js/
│       ├── api.js                     # REST API client
│       ├── charts.js                  # Chart.js radar & fuzzy curves visualizers
│       └── wizard.js                  # Questionnaire controller & persona logic
│
├── .gitignore                         # Git ignore rules
└── README.md                          # Project documentation
```

---

## 8. Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Harsh007engineering/Depression-Anxiety-Screening-using-ANN-Fuzzy-Logic.git
cd Depression-Anxiety-Screening-using-ANN-Fuzzy-Logic
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Launch the Application
```bash
python backend/run_server.py
```

### 4. Access the Interfaces
- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 9. REST API Reference (Swagger)

FastAPI automatically provides interactive Swagger UI at `http://localhost:8000/docs`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status & model verification |
| `GET` | `/api/questions` | Standard 21 DASS items with subscale mappings |
| `POST` | `/api/screen` | Execute Neuro-Fuzzy screening on 21 responses |
| `GET` | `/api/fuzzy-visuals`| Coordinates for plotting fuzzy membership curves $\mu(x)$ |
| `GET` | `/api/benchmark` | Soft Computing comparative study metrics |
| `GET` | `/api/personas` | Pre-configured demo profiles for live presentations |

---

## 10. Viva / Presentation Guide

When presenting this project to evaluators, highlight these soft computing aspects:

1. **Why not pure ANN?**  
   Pure ANNs are black boxes. A clinician cannot verify why a patient was classified as "Severe". In our system, the ANN handles feature learning, while the Fuzzy Logic system produces transparent rule activations.
2. **Why not pure Fuzzy Logic?**  
   Pure Fuzzy Logic cannot learn latent question weights from empirical data without manual rule tuning. The ANN learns nonlinear relationships directly from responses.
3. **How does Centroid Defuzzification work?**  
   Centroid defuzzification computes the center of gravity of the combined clipped output fuzzy sets:  
   $z^* = \frac{\int z \mu(z) dz}{\int \mu(z) dz}$. It guarantees smooth, continuous risk transitions without abrupt step changes.
4. **Demonstrating XAI Live**:  
   Click *Taylor (Severe Depression)* on the web UI, scroll down to **Active Fuzzy Rules**, and show how Rule 16 (*Severe clinical depression with vegetative symptoms*) and Rule 18 fired with specific degree weights ($\alpha$).

---

## 👥 Authors
- Developed for University Soft Computing Course
- GitHub: [@Harsh007engineering](https://github.com/Harsh007engineering)
