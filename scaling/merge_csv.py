"""Merge transfer.csv files made on different computers (e.g. --runs 0-4 on one PC, --runs 5-9 on
another) into one file, keeping each (candidate, scale, seed, drift) row once.

    python scaling/merge_csv.py results/scaling/transfer_all/transfer.csv results/scaling/transfer_a/transfer.csv results/scaling/transfer_b/transfer.csv
then:
    python scaling/transfer.py --summarize results/scaling/transfer_all/transfer.csv
"""
import csv
import os
import sys

if len(sys.argv) < 3:
    raise SystemExit(__doc__)
out, ins = sys.argv[1], sys.argv[2:]
seen, rows, fields = set(), [], None
for path in ins:
    with open(path) as f:
        r = csv.DictReader(f)
        fields = fields or r.fieldnames
        for row in r:
            key = (row["candidate"], row["scale"], row["seed"], row.get("drift", ""), row.get("slip", ""),
                   row.get("train_slip", ""))
            if key not in seen:
                seen.add(key)
                rows.append(row)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows from {len(ins)} files -> {out}")
