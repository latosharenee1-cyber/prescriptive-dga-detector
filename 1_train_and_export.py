import argparse
import json
import math
import os
from typing import Dict, List

import pandas as pd
import h2o
from h2o.automl import H2OAutoML


<<<<<<< HEAD
def entropy(s: str) -> float:
=======
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

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["length"] = out["domain"].astype(str).str.len()
    out["entropy"] = out["domain"].astype(str).apply(entropy)
    return out[["length", "entropy", "label"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DGA detector with H2O AutoML and export artifacts")
    parser.add_argument("--csv", default="data/dga_labeled.csv", help="Path to labeled CSV with columns: domain,label")
    parser.add_argument("--outdir", default="model", help="Output directory for artifacts")
    parser.add_argument("--max_runtime_secs", type=int, default=120, help="AutoML max runtime in seconds")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_raw = pd.read_csv(args.csv)
    required_cols = {"domain", "label"}
    missing = required_cols.difference(df_raw.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = build_features(df_raw)

    bg = {
        "mean_length": float(df["length"].mean()),
        "mean_entropy": float(df["entropy"].mean()),
        "std_length": float(df["length"].std(ddof=0)),
        "std_entropy": float(df["entropy"].std(ddof=0)),
    }
    with open(os.path.join(args.outdir, "background_stats.json"), "w", encoding="utf-8") as f:
        json.dump(bg, f, indent=2)

    schema = {"features": ["length", "entropy"], "label": "label"}
    with open(os.path.join(args.outdir, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    h2o.init()
    try:
        hf = h2o.H2OFrame(df)
        y = "label"
        x: List[str] = ["length", "entropy"]
        hf[y] = hf[y].asfactor()

        aml = H2OAutoML(max_runtime_secs=args.max_runtime_secs, seed=42, exclude_algos=["GLM"])
        aml.train(x=x, y=y, training_frame=hf)

        leader = aml.leader
        print("Leader model:", leader.model_id)

        bin_path = h2o.save_model(leader, path=args.outdir, force=True)
        print("Saved binary model:", bin_path)

        mojo_path = leader.download_mojo(path=args.outdir)
        print("Saved MOJO:", mojo_path)

        artifacts = {
            "binary_model_path": bin_path,
            "mojo_path": mojo_path,
            "features": x,
            "label": y,
        }
        with open(os.path.join(args.outdir, "artifacts.json"), "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2)
    finally:
        h2o.shutdown(prompt=False)
=======

def compute_features(df: pd.DataFrame, domain_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    doms = df[domain_col].astype(str).fillna("")
    out["length"] = doms.str.len()
    out["entropy"] = doms.apply(shannon_entropy).astype(float)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv", default=os.environ.get("DATA_CSV", "data/dga_labeled.csv")
    )
    parser.add_argument("--target_col", default=os.environ.get("TARGET_COL", "label"))
    parser.add_argument("--domain_col", default=os.environ.get("DOMAIN_COL", "domain"))
    parser.add_argument(
        "--max_runtime_secs",
        type=int,
        default=int(os.environ.get("MAX_RUNTIME_SECS", "180")),
    )
    args = parser.parse_args()

    # Load raw data
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(
            f"Could not find dataset at {args.data_csv}. Place a CSV at that path."
        )

    raw = pd.read_csv(args.data_csv)
    if args.domain_col not in raw.columns:
        raise ValueError(
            f"Column {args.domain_col} not found in CSV. Columns present: {list(raw.columns)}"
        )

    if args.target_col not in raw.columns:
        raise ValueError(
            f"Column {args.target_col} not found in CSV. Columns present: {list(raw.columns)}"
        )

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
        exclude_algos=None,
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
        mojo_path = h2o.download_mojo(
            model=leader, path="model", get_genmodel_jar=False
        )
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
        print(
            "MOJO export skipped due to H2O build limitations. Fallback model will be saved."
        )

    # Save binary model as fallback
    bin_dir = h2o.save_model(model=leader, path="model", force=True)
    print("Saved binary model to", bin_dir)

    # Save feature schema
    schema = {"features": x_cols, "target": y_col, "version": 1}
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
        print(
            "Artifacts ready. Use 2_analyze_domain.py which will load the binary model as a fallback."
        )
>>>>>>> fbbd260 (Clean Ruff issues: dataset builder, training, analyzer)



if __name__ == "__main__":
    main()
