import argparse
import json
import math
import os
from typing import Dict

import pandas as pd
import h2o


<<<<<<< HEAD
def entropy(s: str) -> float:
=======
# Generative AI client
try:
    import google.generativeai as genai
except Exception:
    genai = None  # Optional, will handle gracefully


# -----------------------------
# Feature functions
# -----------------------------
def shannon_entropy(s: str) -> float:
>>>>>>> fbbd260 (Clean Ruff issues: dataset builder, training, analyzer)
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(s)
    ent = 0.0
    for c in freq.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

<<<<<<< HEAD

def build_features_for_domain(domain: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "length": [len(str(domain))],
            "entropy": [entropy(str(domain))],
        }
    )

=======

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
        if (
            name.startswith("AutoML")
            or name.startswith("GBM")
            or name.startswith("XGBoost")
            or name.startswith("StackedEnsemble")
        ):
            bin_candidates.append(os.path.join("model", name))
    bin_candidates.sort()
>>>>>>> fbbd260 (Clean Ruff issues: dataset builder, training, analyzer)

def load_artifacts(model_dir: str) -> Dict[str, str]:
    path = os.path.join(model_dir, "artifacts.json")
    if not os.path.exists(path):
        return {
            "binary_model_path": "",
            "mojo_path": os.path.join(model_dir, "DGA_Leader.zip"),
            "features": ["length", "entropy"],
            "label": "label",
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

<<<<<<< HEAD
=======
    # Try MOJO
    if os.path.exists(mojo_path):
        try:
            mojo_key = h2o.import_mojo(mojo_path)
            model = H2OGenericEstimator(model_key=mojo_key)
            # Train with empty x on some frame so H2O can complete model build
            # We create a tiny frame with the expected columns
            tmp = h2o.H2OFrame(pd.DataFrame({"length": [0.0], "entropy": [0.0]}))
            model.train(training_frame=tmp)
            print("Loaded MOJO model from", mojo_path)
            return model
        except Exception as e:
            print("Could not import MOJO in this H2O build:", e)
>>>>>>> fbbd260 (Clean Ruff issues: dataset builder, training, analyzer)

def to_structured_summary(domain: str, prob_dga: float, contrib: Dict[str, float]) -> str:
    parts = [
        f"Domain: {domain}",
        f"Probability DGA: {prob_dga:.3f}",
        "Top contributing features:",
    ]
    for k, v in contrib.items():
        parts.append(f"  - {k}: {v:+.4f}")
    return "\n".join(parts)

<<<<<<< HEAD

def generate_playbook(summary: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    prompt = (
        "You are a security incident responder. Based on the following domain risk summary, "
        "produce a concise, stepwise playbook with containment, investigation, and escalation steps.\n\n"
        f"{summary}"
    )
    if not api_key:
=======
    raise RuntimeError(
        "No usable model was found in the model directory. Train first with 1_train_and_export.py."
    )


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
            pred = preds.get("predict", pd.Series([0] * len(preds))).astype(str)
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
    bg = np.column_stack(
        [
            rng.normal(mu_len, max(std_len, 1e-3), size=n),
            rng.normal(mu_ent, max(std_ent, 1e-3), size=n),
        ]
    )
    # clip to sensible ranges
    bg[:, 0] = np.clip(bg[:, 0], 1, 100)
    bg[:, 1] = np.clip(bg[:, 1], 0, 6)
    return bg


# -----------------------------
# GenAI playbook
# -----------------------------
def make_playbook(summary: dict) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key or genai is None:
>>>>>>> fbbd260 (Clean Ruff issues: dataset builder, training, analyzer)
        return (
            f"{summary}\n\n"
            "Playbook:\n"
            "1. Containment: Add domain to DNS blocklist or sinkhole. Isolate affected hosts.\n"
            "2. Investigation: Search DNS, proxy, and EDR telemetry for related lookups. Check time windows and host overlap.\n"
            "3. Remediation: Clear scheduled tasks, kill processes, reset credentials if exfiltration suspected.\n"
            "4. Escalation: Notify incident response. Preserve logs and forensics artifacts.\n"
            "5. Monitoring: Set up alerts for similar entropy and length patterns in outbound DNS."
        )
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        return text or "No playbook text returned."
    except Exception as exc:  # pragma: no cover
        return (
            f"{summary}\n\n"
            "Playbook generation failed, using fallback.\n"
            "1. Containment: Add domain to DNS blocklist or sinkhole. Isolate affected hosts.\n"
            "2. Investigation: Search DNS, proxy, and EDR telemetry for related lookups. Check time windows and host overlap.\n"
            "3. Remediation: Clear scheduled tasks, kill processes, reset credentials if exfiltration suspected.\n"
            "4. Escalation: Notify incident response. Preserve logs and forensics artifacts.\n"
            f"(Reason: {exc})"
        )


<<<<<<< HEAD
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a domain and generate a prescriptive playbook")
    parser.add_argument("--domain", required=True, help="Domain to analyze, e.g., example.com")
    parser.add_argument("--model_dir", default="model", help="Directory containing exported model artifacts")
=======
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
            text = (
                "LLM did not return text. Here is the summary instead:\n"
                + json.dumps(summary, indent=2)
            )
    return text


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Domain to analyze")
>>>>>>> fbbd260 (Clean Ruff issues: dataset builder, training, analyzer)
    args = parser.parse_args()

    feats_df = build_features_for_domain(args.domain)
    artifacts = load_artifacts(args.model_dir)
    mojo_path = artifacts.get("mojo_path", "")
    bin_path = artifacts.get("binary_model_path", "")

    h2o.init()
    try:
        contributions: Dict[str, float] = {}
        prob_dga = 0.0

        if bin_path and os.path.exists(bin_path):
            model = h2o.load_model(bin_path)
            hf = h2o.H2OFrame(feats_df)
            pred = model.predict(hf)
            prob_dga = float(pred.as_data_frame()["p1"].iloc[0])

            contrib_hf = model.predict_contributions(hf)
            contrib_df = contrib_hf.as_data_frame()
            for col in ["length", "entropy"]:
                if col in contrib_df.columns:
                    contributions[col] = float(contrib_df[col].iloc[0])

        elif mojo_path and os.path.exists(mojo_path):
            model = h2o.import_mojo(mojo_path)  # type: ignore[attr-defined]
            hf = h2o.H2OFrame(feats_df)
            pred = model.predict(hf)
            prob_dga = float(pred.as_data_frame()["p1"].iloc[0])

        else:
            raise FileNotFoundError(
                f"Neither binary model nor MOJO found. Looked for:\n  {bin_path}\n  {mojo_path}"
            )

        summary = to_structured_summary(args.domain, prob_dga, contributions)
        playbook = generate_playbook(summary)
        print(playbook)
    finally:
        h2o.shutdown(prompt=False)

<<<<<<< HEAD
=======
    # Build SHAP explanation with a model agnostic KernelExplainer
    bg = build_background("model/background_stats.json", n=64)
    f = predict_proba_fn(model)
    explainer = shap.KernelExplainer(f, bg)
    shap_values = explainer.shap_values(X[features].values, nsamples=200)

    # KernelExplainer returns arrays
    sv = shap_values if isinstance(shap_values, np.ndarray) else np.array(shap_values)
    sv = sv.reshape(1, -1)
    contribs = [
        {"feature": feat, "value": float(X.iloc[0][feat]), "shap": float(sv[0, i])}
        for i, feat in enumerate(features)
    ]
    contribs_sorted = sorted(contribs, key=lambda d: abs(d["shap"]), reverse=True)

    expected_value = (
        float(explainer.expected_value)
        if hasattr(explainer, "expected_value")
        else None
    )

    summary = {
        "domain": args.domain,
        "predicted_class": predicted_class,
        "prob_legit": round(p0, 4),
        "prob_dga": round(p1, 4),
        "features": contribs_sorted,
        "expected_value": expected_value,
        "notes": "Positive class is DGA. Features are length and entropy.",
    }

    playbook = make_playbook(summary)

    print("\n=== Prescriptive DGA Detector ===")
    print("Domain:", args.domain)
    print(
        f"Prediction: {predicted_class}  P(DGA)={summary['prob_dga']}  P(Legit)={summary['prob_legit']}"
    )
    print("\nExplanation features ordered by impact:")
    for c in contribs_sorted:
        print(f"  {c['feature']}: value={c['value']:.4f}, shap={c['shap']:.5f}")
    print("\n--- Playbook ---")
    print(playbook)
>>>>>>> fbbd260 (Clean Ruff issues: dataset builder, training, analyzer)


if __name__ == "__main__":
    main()
