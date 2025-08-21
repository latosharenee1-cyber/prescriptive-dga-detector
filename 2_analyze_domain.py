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

# 2_analyze_domain.py
import argparse
import json
import math
import os
from typing import Dict

import pandas as pd
import h2o


def entropy(s: str) -> float:
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


def build_features_for_domain(domain: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "length": [len(str(domain))],
            "entropy": [entropy(str(domain))],
        }
    )


def load_artifacts(model_dir: str) -> Dict[str, str]:
    path = os.path.join(model_dir, "artifacts.json")
    if not os.path.exists(path):
        # Fallback to conventional names
        return {
            "binary_model_path": "",
            "mojo_path": os.path.join(model_dir, "DGA_Leader.zip"),
            "features": ["length", "entropy"],
            "label": "label",
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_structured_summary(domain: str, prob_dga: float, contrib: Dict[str, float]) -> str:
    parts = [
        f"Domain: {domain}",
        f"Probability DGA: {prob_dga:.3f}",
        "Top contributing features:",
    ]
    for k, v in contrib.items():
        parts.append(f"  - {k}: {v:+.4f}")
    return "\n".join(parts)


def generate_playbook(summary: str) -> str:
    # Optional: use Google Generative AI if key is present
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    prompt = (
        "You are a security incident responder. Based on the following domain risk summary, "
        "produce a concise, stepwise playbook with containment, investigation, and escalation steps.\n\n"
        f"{summary}"
    )
    if not api_key:
        # Safe fallback if no API key configured
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
        # Fallback if API errors
        return (
            f"{summary}\n\n"
            "Playbook generation failed, using fallback.\n"
            "1. Containment: Add domain to DNS blocklist or sinkhole. Isolate affected hosts.\n"
            "2. Investigation: Search DNS, proxy, and EDR telemetry for related lookups. Check time windows and host overlap.\n"
            "3. Remediation: Clear scheduled tasks, kill processes, reset credentials if exfiltration suspected.\n"
            "4. Escalation: Notify incident response. Preserve logs and forensics artifacts.\n"
            f"(Reason: {exc})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a domain and generate a prescriptive playbook")
    parser.add_argument("--domain", required=True, help="Domain to analyze, e.g., example.com")
    parser.add_argument("--model_dir", default="model", help="Directory containing exported model artifacts")
    args = parser.parse_args()

    feats_df = build_features_for_domain(args.domain)

    artifacts = load_artifacts(args.model_dir)
    mojo_path = artifacts.get("mojo_path", "")
    bin_path = artifacts.get("binary_model_path", "")

    h2o.init()
    try:
        # Prefer binary model for contributions (SHAP-like). If not available, load MOJO for prediction only.
        contributions = {}
        prob_dga = 0.0

        if bin_path and os.path.exists(bin_path):
            model = h2o.load_model(bin_path)
            hf = h2o.H2OFrame(feats_df)
            pred = model.predict(hf)
            # Binary classifier: get probability of class "1"
            prob_dga = float(pred.as_data_frame()["p1"].iloc[0])

            # SHAP-like contributions (bias term plus feature contributions)
            contrib_hf = model.predict_contributions(hf)
            contrib_df = contrib_hf.as_data_frame()
            # Keep only our known features
            for col in ["length", "entropy"]:
                if col in contrib_df.columns:
                    contributions[col] = float(contrib_df[col].iloc[0])

        elif mojo_path and os.path.exists(mojo_path):
            # MOJO prediction path (no contributions)
            # h2o.import_mojo available on recent H2O; fallback to Python mojo pipeline if configured.
            # Best-effort approach: load MOJO as model and predict.
            model = h2o.import_mojo(mojo_path)  # type: ignore[attr-defined]
            hf = h2o.H2OFrame(feats_df)
            pred = model.predict(hf)
            prob_dga = float(pred.as_data_frame()["p1"].iloc[0])
            # No contributions for MOJO; leave contributions empty.
        else:
            raise FileNotFoundError(
                f"Neither binary model nor MOJO found. Looked for:\n  {bin_path}\n  {mojo_path}"
            )

        # Build summary and generate prescriptive playbook
        summary = to_structured_summary(args.domain, prob_dga, contributions)
        playbook = generate_playbook(summary)
        print(playbook)

    finally:
        h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()
