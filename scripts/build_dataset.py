import argparse
import csv
import os
import random
from typing import List

COMMON_COLS = ["domain", "domains", "url", "host", "hostname", "fqdn"]


def normalize_domain(s: str) -> str:
    """Lowercase and strip dots and whitespace."""
    return (s or "").strip().lower().strip(".")


def read_domains(path: str) -> List[str]:
    """
    Read domains from a .txt or .csv file.

    - .txt: one domain per line
    - .csv: tries common column names; falls back to first column
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    domains: List[str] = []
    seen = set()
    name = os.path.basename(path).lower()

    if name.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                d = normalize_domain(line)
                if d and "." in d and d not in seen:
                    seen.add(d)
                    domains.append(d)

    elif name.endswith(".csv"):
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = ""
                for col in COMMON_COLS:
                    if col in row:
                        d = normalize_domain(row.get(col, ""))
                        if d:
                            break
                if not d and row:
                    first_key = next(iter(row.keys()))
                    d = normalize_domain(row.get(first_key, ""))

                if d and "." in d and d not in seen:
                    seen.add(d)
                    domains.append(d)
    else:
        raise ValueError("Only .txt and .csv are supported")

    return domains


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled DGA dataset")
    parser.add_argument("--positives", required=True, help="Path to DGA domains (.txt or .csv)")
    parser.add_argument("--negatives", required=True, help="Path to legit domains (.txt or .csv)")
    parser.add_argument("--sample_pos", type=int, default=0, help="Optional sample size for positives")
    parser.add_argument("--sample_neg", type=int, default=0, help="Optional sample size for negatives")
    args = parser.parse_args()

    pos = read_domains(args.positives)
    neg = read_domains(args.negatives)

    if args.sample_pos and len(pos) > args.sample_pos:
        random.seed(42)
        pos = random.sample(pos, args.sample_pos)

    if args.sample_neg and len(neg) > args.sample_neg:
        random.seed(42)
        neg = random.sample(neg, args.sample_neg)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "dga_labeled.csv")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "label"])
        writer.writeheader()
        for d in pos:
            writer.writerow({"domain": d, "label": 1})
        for d in neg:
            writer.writerow({"domain": d, "label": 0})

    print(f"Wrote {len(pos) + len(neg)} rows to {out_path}")


if __name__ == "__main__":
    main()
