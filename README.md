# Prescriptive DGA Detector

End-to-end tool that classifies a domain as **Legit** or **DGA**, explains *why* using **SHAP**, and generates a **prescriptive incident-response playbook** using **Google Gemini**.

---

## Goals

1. **Rapid model building** with H2O AutoML.  
2. **Explainability** with SHAP (model-agnostic).  
3. **Prescription** via Generative AI (playbook tailored to the prediction & SHAP).

---

## Architecture

**Training (`1_train_and_export.py`)**
- Computes simple features: `length`, `entropy`.
- Trains with **H2O AutoML**.
- Exports leader as **MOJO** (`model/DGA_Leader.zip`) *and* saves a **binary fallback**.
- Writes `model/feature_schema.json` and `model/background_stats.json` (for SHAP background).

**Analyzer (`2_analyze_domain.py`)**
- Input: `--domain yourdomain.tld`.
- Loads the **MOJO** (or binary fallback) and predicts P(DGA).
- Explains the decision with **SHAP KernelExplainer**.
- Builds a structured summary and prompts **Gemini** to generate a prescriptive playbook.

---

## Prerequisites

- **Python** ≥ 3.10 (tested on 3.11)
- **Java JDK 17** (required by H2O)
  - Windows (admin CMD):
    ```cmd
    winget install -e --id EclipseAdoptium.Temurin.17.JDK
    ```
  - Verify in a new terminal:
    ```cmd
    java -version
    ```

---

## Setup

### Windows (CMD)
```cmd
cd %USERPROFILE%\Downloads\prescriptive_dga_detector_plus2
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install -r requirements.txt