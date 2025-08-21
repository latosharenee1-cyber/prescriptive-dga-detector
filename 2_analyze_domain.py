"""
2_analyze_domain.py

Main application that accepts a domain name, loads the MOJO leader if available
or the binary model as a fallback, performs prediction, explains the decision with SHAP,
and generates a prescriptive response playbook with Google Generative AI.

Usage
python 2_analyze_domain.py --domain example.com

Environment
GOOGLE_API_KEY must be set to call Generative AI. If it is missing, the script will
print the structured summary and a helpful message instead of calling the API.
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd

import h2o
from h2o.estimators.generic import H2OGenericEstimator

# SHAP for model agnostic explanation
import shap

# Generative AI client
try:
    import google.generativeai as genai
except Exception:
    genai = None  # Optional, will handle gracefully

# -----------------------------
# Feature functions
# -----------------------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return float(ent)

def featurize_one(domain: str) -> pd.DataFrame:
    dom = str(domain or "")
    row = {
        "length": len(dom),
        "entropy": shannon_entropy(dom),
    }
    return pd.DataFrame([row])

# -----------------------------
# H2O model loader
# -----------------------------
def load_model():
    """
    Try to load the MOJO first using H2OGenericEstimator with import_mojo.
    If that fails, load the binary model directory created by h2o.save_model.
    """
    mojo_path = "model/DGA_Leader.zip"
    bin_candidates = []

    # Find a saved binary model directory inside model/
    for name in os.listdir("model"):
        if name.startswith("AutoML") or name.startswith("GBM") or name.startswith("XGBoost") or name.startswith("StackedEnsemble"):
            bin_candidates.append(os.path.join("model", name))
    bin_candidates.sort()

    # Start H2O or connect if running
    try:
        h2o.init()
    except Exception:
        # If a server is already running, try connect
        h2o.connect()

    # Try MOJO
    if os.path.exists(mojo_path):
        try:
            mojo_key = h2o.import_mojo(mojo_path)
            model = H2OGenericEstimator(model_key=mojo_key)
            # Train with empty x on some frame so H2O can complete model build
            # We create a tiny frame with the expected columns
            tmp = h2o.H2OFrame(pd.DataFrame({"length":[0.0], "entropy":[0.0]}))
            model.train(training_frame=tmp)
            print("Loaded MOJO model from", mojo_path)
            return model
        except Exception as e:
            print("Could not import MOJO in this H2O build:", e)

    # Fallback to binary model
    for path in bin_candidates:
        try:
            m = h2o.load_model(path)
            print("Loaded binary model from", path)
            return m
        except Exception as e:
            print("Tried binary model", path, "but got error:", e)

    raise RuntimeError("No usable model was found in the model directory. Train first with 1_train_and_export.py.")

# -----------------------------
# SHAP explanation
# -----------------------------
def predict_proba_fn(h2o_model):
    """
    Return a function that maps a numpy array shaped (n, 2) of features
    to probabilities for the positive class.
    """
    def _predict(X: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X, columns=["length", "entropy"])
        hf = h2o.H2OFrame(df)
        preds = h2o_model.predict(hf).as_data_frame()
        # Expect columns like ['predict', 'p0', 'p1'] where p1 is the positive class probability
        if "p1" in preds.columns:
            return preds["p1"].values
        elif "DGA" in preds.columns:  # in case class name is string
            return preds["DGA"].values
        else:
            # If only predict column is available, map to 1.0 for label 1
            pred = preds.get("predict", pd.Series([0]*len(preds))).astype(str)
            return (pred == "1").astype(float).values
    return _predict

def build_background(stats_path: str, n: int = 64) -> np.ndarray:
    """
    Build a small background sample for KernelExplainer from saved training stats.
    """
    mu_len = 12.0
    std_len = 5.0
    mu_ent = 3.0
    std_ent = 0.8

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mu_len = float(stats["length"]["mean"])
        std_len = float(stats["length"]["std"] or 1.0)
        mu_ent = float(stats["entropy"]["mean"])
        std_ent = float(stats["entropy"]["std"] or 0.5)
    except Exception:
        pass

    rng = np.random.default_rng(42)
    bg = np.column_stack([
        rng.normal(mu_len, max(std_len, 1e-3), size=n),
        rng.normal(mu_ent, max(std_ent, 1e-3), size=n),
    ])
    # clip to sensible ranges
    bg[:,0] = np.clip(bg[:,0], 1, 100)
    bg[:,1] = np.clip(bg[:,1], 0, 6)
    return bg

# -----------------------------
# GenAI playbook
# -----------------------------
def make_playbook(summary: dict) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key or genai is None:
        return (
            "Generative AI is not configured. Set GOOGLE_API_KEY and install google-generativeai to enable playbook creation.\n\n"
            + json.dumps(summary, indent=2)
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are a senior incident responder. Create a short prescriptive playbook for the SOC based on this classifier decision.
Keep it clear and immediately actionable.

Structured summary
{json.dumps(summary, indent=2)}

Write sections
- Triage and decide
- Collection and containment
- SIEM hunt and detection content
- Network and DNS controls to apply now
- Follow up validation steps
- Communication notes

Keep commands and queries concrete. Use generic query examples for Splunk and Sentinel. Use domain variables where helpful.
"""

    resp = model.generate_content(prompt)
    text = ""
    try:
        text = resp.candidates[0].content.parts[0].text  # best effort extraction
    except Exception:
        try:
            text = resp.text
        except Exception:
            text = "LLM did not return text. Here is the summary instead:\n" + json.dumps(summary, indent=2)
    return text

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Domain to analyze")
    args = parser.parse_args()

    # Featurize the input
    X = featurize_one(args.domain)
    features = ["length", "entropy"]

    # Load H2O model
    model = load_model()

    # Predict class and probability
    hf = h2o.H2OFrame(X[features])
    pred = model.predict(hf).as_data_frame()
    # Normalize outputs
    if "p1" in pred.columns:
        p1 = float(pred.loc[0, "p1"])
        p0 = float(pred.loc[0, "p0"])
        label = str(pred.loc[0, "predict"])
    else:
        # Fallback
        if "predict" in pred.columns:
            label = str(pred.loc[0, "predict"])
            p1 = 1.0 if label == "1" else 0.0
            p0 = 1.0 - p1
        else:
            # last resort
            p1 = float(pred.iloc[0].mean())
            p0 = 1.0 - p1
            label = "1" if p1 >= 0.5 else "0"

    predicted_class = "DGA" if label in ("1", "DGA", "true") else "Legit"

    # Build SHAP explanation with a model agnostic KernelExplainer
    bg = build_background("model/background_stats.json", n=64)
    f = predict_proba_fn(model)
    explainer = shap.KernelExplainer(f, bg)
    shap_values = explainer.shap_values(X[features].values, nsamples=200)

    # KernelExplainer returns arrays
    sv = shap_values if isinstance(shap_values, np.ndarray) else np.array(shap_values)
    sv = sv.reshape(1, -1)
    contribs = [{ "feature": feat, "value": float(X.iloc[0][feat]), "shap": float(sv[0, i]) } for i, feat in enumerate(features)]
    contribs_sorted = sorted(contribs, key=lambda d: abs(d["shap"]), reverse=True)

    expected_value = float(explainer.expected_value) if hasattr(explainer, "expected_value") else None

    summary = {
        "domain": args.domain,
        "predicted_class": predicted_class,
        "prob_legit": round(p0, 4),
        "prob_dga": round(p1, 4),
        "features": contribs_sorted,
        "expected_value": expected_value,
        "notes": "Positive class is DGA. Features are length and entropy."
    }

    playbook = make_playbook(summary)

    print("\n=== Prescriptive DGA Detector ===")
    print("Domain:", args.domain)
    print(f"Prediction: {predicted_class}  P(DGA)={summary['prob_dga']}  P(Legit)={summary['prob_legit']}")
    print("\nExplanation features ordered by impact:")
    for c in contribs_sorted:
        print(f"  {c['feature']}: value={c['value']:.4f}, shap={c['shap']:.5f}")
    print("\n--- Playbook ---")
    print(playbook)

if __name__ == "__main__":
    main()
