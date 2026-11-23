"""Put several experiments (variants) side by side in one table.

For every results folder and both algorithms it reports:
    - the fresh-seed learning speed of the winners on every maze (from reevaluation.csv),
    - the MAP-Elites map (cells filled, cells whose scheme learns on all seeds),
    - the genes of the winners, and how many winners sit at the outer edge of the range
      (many winners at the edge = the range may still be too narrow for that gene),
    - a test against the first folder (the reference) on the training maze.

Run from the repository folder, after reevaluate_variant.py:
    python variants/compare_variants.py
    python variants/compare_variants.py --dirs results/2026-11-15_laptop results/variants/wide_x2

Default: the saved default experiment (results/2026-11-15_laptop) as reference, then every
folder in results/variants/ that has a reevaluation.csv. Output: results/variants/comparison.csv

A comparison is only fair when the folders used the same budget and number of runs
(both are printed). The unit of the test is the winner (one per run), not the single
training run, so 10 runs per variant give n = 10.
"""
import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

import variant_defs
import evolve

TRAIN_MAZE = "dyna_9x6_3traps"
ALGOS = [("mapelites", "ME best"), ("de", "DE best"), ("random", "RS best")]
EDGE = 0.05          # "at the edge" = within 5% of the range from its outer end
OUTER_IS_HIGH = {"progress"}     # progress grows upwards; trap, wall and step grow downwards


def per_winner_medians(folder):
    """{(maze, candidate): {scheme: (median N90, share learned)}} from reevaluation.csv."""
    raw = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(folder, "reevaluation.csv")) as f:
        for r in csv.DictReader(f):
            raw[(r["maze"], r["candidate"])][r["scheme"]].append(
                (float(r["episodes_to_90"]), r["converged"] == "True"))
    return {k: {s: (statistics.median(x[0] for x in v), statistics.mean(x[1] for x in v))
                for s, v in schemes.items()} for k, schemes in raw.items()}


def mann_whitney(x, y):
    """Two-sided p-value and A12 = chance that a winner of x learns FASTER than one of y."""
    wins = sum((a < b) + 0.5 * (a == b) for a in x for b in y) / (len(x) * len(y))
    try:
        from scipy.stats import mannwhitneyu
        p = mannwhitneyu(x, y, alternative="two-sided").pvalue
    except ImportError:
        p = float("nan")
    return p, wins


def describe(folder):
    runs = variant_defs.load_runs(folder)
    bounds = variant_defs.bounds_of_runs(runs)
    variant_defs.apply_bounds(bounds)
    fresh = per_winner_medians(folder)
    mazes = sorted({m for m, _ in fresh}, key=lambda m: (m != TRAIN_MAZE, m))
    rows = []
    for algo, cand in ALGOS:
        rs = [r for r in runs if r["algo"] == algo]
        if not rs:
            continue
        row = dict(folder=os.path.basename(os.path.normpath(folder)),
                   variant=rs[0].get("variant", "default"), algo=cand.split()[0], runs=len(rs),
                   budget="/".join(sorted({str(r["budget"]) for r in rs})),
                   evolution_fitness=statistics.median(r["best"]["fitness"] for r in rs))
        for m in mazes:
            w = fresh.get((m, cand), {})
            row[f"n90_{m}"] = statistics.median(v[0] for v in w.values()) if w else ""
        w = fresh.get((TRAIN_MAZE, cand), {})
        row["pct_learned_train"] = round(100 * statistics.mean(v[1] for v in w.values()), 1) if w else ""
        row["_train"] = [v[0] for v in w.values()]
        if "archive" in rs[0]:                     # MAP-Elites and random search keep a map
            row["cells_filled"] = statistics.median(len(r["archive"]) for r in rs)
            row["cells_learning"] = statistics.median(
                sum(e["converged"] == len(e["n90"]) for e in r["archive"]) for r in rs)
        else:
            row["cells_filled"] = row["cells_learning"] = ""
        schemes = [evolve.decode(r["best"]["genome"]) for r in rs]
        for i, g in enumerate(evolve.GENES):
            row[f"gene_{g}"] = round(statistics.median(getattr(s, g) for s in schemes), 2)
            if g in variant_defs.REWARD_GENES and bounds[g][0] != bounds[g][1]:
                xs = [r["best"]["genome"][i] for r in rs]
                row[f"edge_{g}"] = sum(x > 1 - EDGE if g in OUTER_IS_HIGH else x < EDGE for x in xs)
            elif g in variant_defs.REWARD_GENES:
                row[f"edge_{g}"] = ""
        rows.append(row)
    return rows, bounds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", help="results folders; the first one is the reference")
    p.add_argument("--out", default=os.path.join(variant_defs.ROOT, "results", "variants", "comparison.csv"))
    a = p.parse_args()
    dirs = a.dirs or ([os.path.join(variant_defs.ROOT, "results", "2026-11-15_laptop")]
                      + sorted(glob.glob(os.path.join(variant_defs.ROOT, "results", "variants", "*", ""))))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "reevaluation.csv"))]
    if not dirs:
        raise SystemExit("no folder with a reevaluation.csv found (run reevaluate_variant.py first)")

    table, ref = [], {}
    for d in dirs:
        rows, bounds = describe(d)
        print(f"{os.path.basename(os.path.normpath(d)):20s} ranges: "
              + ", ".join(f"{g} {lo:g}..{hi:g}" for g, (lo, hi) in bounds.items()))
        for row in rows:
            train = row.pop("_train")
            if d == dirs[0]:
                ref[row["algo"]] = (train, row["budget"], row["runs"])
                row["p_vs_reference"] = row["faster_than_reference"] = ""
            elif row["algo"] in ref and train:
                pv, a12 = mann_whitney(train, ref[row["algo"]][0])
                row["p_vs_reference"], row["faster_than_reference"] = round(pv, 4), round(a12, 2)
                if (row["budget"], row["runs"]) != ref[row["algo"]][1:]:
                    row["p_vs_reference"] = f"{row['p_vs_reference']} (budget or runs differ!)"
            table.append(row)

    def show(title, cols, names):
        print(f"\n{title}")
        width = [max([len(n)] + [len(str(r.get(c, ""))) for r in table]) for c, n in zip(cols, names)]
        print("  ".join(f"{n:>{w}s}" for n, w in zip(names, width)))
        for r in table:
            print("  ".join(f"{str(r.get(c, '')):>{w}s}" for c, w in zip(cols, width)))

    mazes = [c for c in table[0] if c.startswith("n90_")]
    base = ["folder", "algo", "runs", "budget"]
    show("Learning speed of the winners on fresh seeds (median episodes to learn; lower is better)",
         base + ["evolution_fitness"] + mazes + ["pct_learned_train"],
         base + ["evo fitness"] + [m[4:] for m in mazes] + ["% learned"])
    show(f"Against the reference ({os.path.basename(os.path.normpath(dirs[0]))}) on the training maze, "
         "and the MAP-Elites map",
         base + ["p_vs_reference", "faster_than_reference", "cells_filled", "cells_learning"],
         base + ["p value", "chance faster", "cells filled", "cells learning"])
    genes = [f"gene_{g}" for g in evolve.GENES]
    edges = [f"edge_{g}" for g in variant_defs.REWARD_GENES]
    show("Genes of the winners (median) and number of winners at the outer edge of the range",
         base + genes + edges, base + list(evolve.GENES) + [f"edge {g}" for g in variant_defs.REWARD_GENES])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fields = list(dict.fromkeys(k for r in table for k in r))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
