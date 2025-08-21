"""
1_train_and_export.py

End to end trainer for the Prescriptive DGA Detector.
This script trains an H2O AutoML model on simple domain features and exports the leader as a MOJO zip.
It also saves a binary model as a fallback in case your H2O build does not yet support MOJO import in Python.
The feature set is intentionally small so your results are fast and easy to explain.

Expected input
A CSV file with columns:
- domain: the full domain string
- label: target label as 1 for DGA and 0 for legit

You can change column names via environment variables if your file is different.
DATA_CSV points to your dataset path, default data/dga_labeled.csv
TARGET_COL sets the label name, default label
DOMAIN_COL sets the domain name column, default domain

Outputs
- model/DGA_Leader.zip      MOJO file for scoring
- model/leader_bin          H2O binary model directory for fallback loading
- model/feature_schema.json Feature list used during training
- model/background_stats.json  Simple stats used to seed SHAP background

Usage
python 1_train_and_export.py
"""

import os
import math
import json
import argparse
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

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

def compute_features(df: pd.DataFrame, domain_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    doms = df[domain_col].astype(str).fillna("")
    out["length"] = doms.str.len()
    out["entropy"] = doms.apply(shannon_entropy).astype(float)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default=os.environ.get("DATA_CSV", "data/dga_labeled.csv"))
    parser.add_argument("--target_col", default=os.environ.get("TARGET_COL", "label"))
    parser.add_argument("--domain_col", default=os.environ.get("DOMAIN_COL", "domain"))
    parser.add_argument("--max_runtime_secs", type=int, default=int(os.environ.get("MAX_RUNTIME_SECS", "180")))
    args = parser.parse_args()

    # Load raw data
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"Could not find dataset at {args.data_csv}. Place a CSV at that path.")

    raw = pd.read_csv(args.data_csv)
    if args.domain_col not in raw.columns:
        raise ValueError(f"Column {args.domain_col} not found in CSV. Columns present: {list(raw.columns)}")

    if args.target_col not in raw.columns:
        raise ValueError(f"Column {args.target_col} not found in CSV. Columns present: {list(raw.columns)}")

    # Compute features
    X = compute_features(raw, args.domain_col)
    y = raw[args.target_col].astype(int)
    df = X.copy()
    df[args.target_col] = y.values

    # Start H2O
    h2o.init()
    hf = h2o.H2OFrame(df)
    hf[args.target_col] = hf[args.target_col].asfactor()

    x_cols = ["length", "entropy"]
    y_col = args.target_col

    # Split
    train, valid = hf.split_frame(ratios=[0.8], seed=42)

    # AutoML
    aml = H2OAutoML(
        max_runtime_secs=args.max_runtime_secs,
        seed=42,
        balance_classes=True,
        sort_metric="AUC",
        exclude_algos=None
    )
    aml.train(x=x_cols, y=y_col, training_frame=train, validation_frame=valid)

    leader = aml.leader
    print("Leader:", leader.model_id)
    print(leader.model_performance(valid=True))

    # Save artifacts
    os.makedirs("model", exist_ok=True)

    # Save MOJO with robust path handling across H2O versions
    mojo_path = None
    try:
        mojo_path = h2o.download_mojo(model=leader, path="model", get_genmodel_jar=False)
    except Exception:
        try:
            mojo_path = leader.download_mojo(path="model", get_genmodel_jar=False)
        except Exception as e:
            print("Warning: Could not export MOJO with current H2O build:", e)

    # Normalize name to DGA_Leader.zip if we got a mojo
    final_mojo_path = None
    if mojo_path and os.path.exists(mojo_path):
        final_mojo_path = os.path.join("model", "DGA_Leader.zip")
        try:
            if os.path.exists(final_mojo_path):
                os.remove(final_mojo_path)
        except Exception:
            pass
        os.rename(mojo_path, final_mojo_path)
        print("Saved MOJO to", final_mojo_path)
    else:
        print("MOJO export skipped due to H2O build limitations. Fallback model will be saved.")

    # Save binary model as fallback
    bin_dir = h2o.save_model(model=leader, path="model", force=True)
    print("Saved binary model to", bin_dir)

    # Save feature schema
    schema = {
        "features": x_cols,
        "target": y_col,
        "version": 1
    }
    with open("model/feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Save simple background stats for SHAP KernelExplainer
    bg = df[x_cols].describe().to_dict()
    with open("model/background_stats.json", "w", encoding="utf-8") as f:
        json.dump(bg, f, indent=2)

    print("\nTraining complete.")
    if final_mojo_path:
        print("Artifacts ready. Use 2_analyze_domain.py to score a domain.")
    else:
        print("Artifacts ready. Use 2_analyze_domain.py which will load the binary model as a fallback.")

    h2o.shutdown(prompt=False)

if __name__ == "__main__":
    main()
