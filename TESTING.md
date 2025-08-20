# TESTING

This document shows how the end-to-end pipeline was validated:
**H2O AutoML → SHAP → Gemini playbook**.

---

## Environment

- OS: Windows 10 (10.0.26100.4946)
- Python: 3.11 (venv)
- Java: Temurin OpenJDK 17.0.16+8 (`java -version`)
- H2O: 3.44.0.3 (local server on `http://127.0.0.1:54321`)
- LLM: Gemini **1.5-flash** via `google-generativeai`

Screenshots collected:
1. `01_virtualenv_activated.png`
2. `02_requirements_installed.png`
3. `02b_dataset_built.png`
4. `03_model_trained_and_exported.png`
5. `04_domain_analysis_playbook.png` (legit)
6. `04_domain_analysis_playbook_dga.png` (DGA-like)

---

## Dataset build & sanity check

**Build the labeled CSV**
```cmd
python scripts\build_dataset.py --positives data\dga_feed.txt --negatives data\legit_tranco.txt

## Observed

Wrote data/dga_labeled.csv: 500 positives, 500 negatives, total 1000

