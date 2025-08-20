"""
scripts/build_dataset.py

Create data/dga_labeled.csv from two local files that you provide:
- positives: DGA domains (text or CSV)
- negatives: legit domains (text or CSV)

The input can be:
- a .txt file with one domain per line
- a .csv file with a column named "domain" (case insensitive)

Usage
python scripts/build_dataset.py --positives path/to/dga.txt --negatives path/to/legit.csv

Optional
--sample_pos N   randomly sample N positives
--sample_neg N   randomly sample N negatives

Output
data/dga_labeled.csv with columns: domain,label
"""

import os
import csv
import argparse
import random

def read_domains(path):
    domains = []
    low = set()
    name = os.path.basename(path).lower()
    try:
        if name.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    d = line.strip().lower().strip(".")
                    if d and "." in d and d not in low:
                        low.add(d); domains.append(d)
        elif name.endswith(".csv"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                # try several common column names
                candidates = [c for c in reader.fieldnames or [] if c and c.lower() in ("domain","domains","fqdn")]
                if not candidates:
                    raise ValueError(f"No 'domain' column found in {path}. Columns: {reader.fieldnames}")
                col = candidates[0]
                for row in reader:
                    d = (row.get(col) or "").strip().lower().strip(".")
                    if d and "." in d and d not in low:
                        low.add(d); domains.append(d)
        else:
            raise ValueError("Only .txt and .csv are supported")
    except Exception as e:
        raise
    return domains

def write_labeled(domains, label, writer):
    for d in domains:
        writer.writerow({"domain": d, "label": label})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", required=True, help="File with DGA domains")
    ap.add_argument("--negatives", required=True, help="File with legit domains")
    ap.add_argument("--sample_pos", type=int, default=None)
    ap.add_argument("--sample_neg", type=int, default=None)
    args = ap.parse_args()

    pos = read_domains(args.positives)
    neg = read_domains(args.negatives)

    if args.sample_pos and len(pos) > args.sample_pos:
        random.seed(42); pos = random.sample(pos, args.sample_pos)
    if args.sample_neg and len(neg) > args.sample_neg:
        random.seed(42); neg = random.sample(neg, args.sample_neg)

    os.makedirs("data", exist_ok=True)
    out_path = "data/dga_labeled.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain","label"])
        w.writeheader()
        write_labeled(pos, 1, w)
        write_labeled(neg, 0, w)

    print(f"Wrote {out_path}: {len(pos)} positives, {len(neg)} negatives, total {len(pos)+len(neg)}")

if __name__ == "__main__":
    main()
