Set-Content -LiteralPath "1_train_and_export.py" -NoNewline -Value @'
import argparse
import json
import math
import os
from typing import Dict, List

import pandas as pd
import h2o
from h2o.automl import H2OAutoML


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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["length"] = out["domain"].astype(str).str.len()
    out["entropy"] = out["domain"].astype(str).apply(entropy)
    return out[["length", "entropy", "label"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DGA detector with H2O AutoML and export artifacts"
    )
    parser.add_argument(
        "--csv",
        default="data/dga_labeled.csv",
        help="Path to labeled CSV with columns: domain,label",
    )
    parser.add_argument(
        "--outdir", default="model", help="Output directory for artifacts"
    )
    parser.add_argument(
        "--max_runtime_secs",
        type=int,
        default=120,
        help="AutoML max runtime in seconds",
    )
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
    with open(
        os.path.join(args.outdir, "background_stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(bg, f, indent=2)

    schema = {"features": ["length", "entropy"], "label": "label"}
    with open(
        os.path.join(args.outdir, "feature_schema.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(schema, f, indent=2)

    h2o.init()
    try:
        hf = h2o.H2OFrame(df)
        y = "label"
        x: List[str] = ["length", "entropy"]
        hf[y] = hf[y].asfactor()

        aml = H2OAutoML(
            max_runtime_secs=args.max_runtime_secs, seed=42, exclude_algos=["GLM"]
        )
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
        with open(
            os.path.join(args.outdir, "artifacts.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(artifacts, f, indent=2)
    finally:
        h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()
