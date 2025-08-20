"""
scripts/preview_features.py

Quick sanity check on the dataset. Prints basic stats and a small sample
with length and entropy so you can see rough separation before training.

Usage
python scripts/preview_features.py --data data/dga_labeled.csv
"""

import csv
import argparse
import math
from collections import Counter

def entropy(s):
    counts = Counter(s)
    n = len(s)
    if n == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * (math.log(p, 2))
    return float(ent)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dga_labeled.csv")
    ap.add_argument("--n", type=int, default=8, help="rows to preview per class")
    args = ap.parse_args()

    pos, neg = [], []
    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            d = (row.get("domain") or "").strip()
            y = int(row.get("label") or 0)
            if y == 1:
                pos.append(d)
            else:
                neg.append(d)

    print(f"Rows: pos={len(pos)} neg={len(neg)} total={len(pos)+len(neg)}")
    print("\nSample positives:")
    for d in pos[:args.n]:
        print(f"  {d:35s}  len={len(d):3d}  ent={entropy(d):.3f}")
    print("\nSample negatives:")
    for d in neg[:args.n]:
        print(f"  {d:35s}  len={len(d):3d}  ent={entropy(d):.3f}")

if __name__ == "__main__":
    main()
