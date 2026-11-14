"""Compare MAP-Elites, Differential Evolution and the hand-made baselines.

Reads the outputs of run_evolution.py and reevaluate.py and produces:

  1. results/analysis/summary.csv + printed table
       per maze and candidate: median episodes to 90% (N90), IQR, share of seeds that learned
  2. results/analysis/stats.csv + printed table
       Mann-Whitney U tests with Holm correction and Vargha-Delaney A12 effect size for
         ME vs DE                       (unit = one evolutionary run: mean N90 over the fresh seeds)
         ME vs each baseline, DE vs each baseline   (unit = one training seed)
  3. results/analysis/boxplot_<maze>.png      N90 of every candidate on that maze
  4. results/analysis/convergence.png         best fitness vs evaluations, ME vs DE (median + IQR)

Examples:
    python reevaluate.py --runs results/evolution --seeds 30       # first
    python analyze.py                                              # main 6 baselines, 5 mazes -> results/analysis/
    python analyze.py --all                                        # everything -> results/analysis_all/ (supplementary)
    python analyze.py --baselines A1 B1_immediate kenton           # a custom selection
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import warnings

import numpy as np
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
import matplotlib.pyplot as plt

ALPHA = 0.05
# Main baselines shown in the tables, tests and plots (all of them stay in reevaluation.csv).
# Short prefixes are fine: "A1" matches "A1_immediate_reward_only".
MAIN_BASELINES = ["A1", "A2", "B1_immediate", "B2", "sparse_dyna", "russell_norvig_matched"]
MAIN_MAZES = ["dyna_9x6_3traps", "dyna_9x6_2traps", "maze_8x8", "maze_10x10", "lava_7x5"]  # dyna_9x6: too easy


def a12(x, y):
    """Vargha-Delaney A12 = P(a value of x is SMALLER than one of y), ties count 1/2.
    Here smaller N90 = better, so A12 > 0.5 means x learns faster than y.
    Rule of thumb: |A12 - 0.5| < 0.06 negligible, < 0.14 small, < 0.21 medium, otherwise large."""
    x, y = np.asarray(x), np.asarray(y)
    less = (x[:, None] < y[None, :]).sum()
    ties = (x[:, None] == y[None, :]).sum()
    return (less + 0.5 * ties) / (len(x) * len(y))


def holm(pvals):
    """Holm-Bonferroni correction. Returns adjusted p-values in the original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def keep_baseline(name, wanted):
    return wanted is None or any(name == w or name.startswith(w) for w in wanted)


def load(path, baselines=None, mazes=None):
    data = defaultdict(lambda: defaultdict(list))        # maze -> candidate -> list of N90
    runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # maze -> algo -> run -> N90s
    conv = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            if mazes is not None and r["maze"] not in mazes:
                continue
            n90 = float(r["episodes_to_90"])
            if r["candidate"] == "baseline":
                name = r["scheme"]
                if not keep_baseline(name, baselines):
                    continue
            else:
                name = r["candidate"]                      # "ME best" / "DE best" (all runs pooled)
                runs[r["maze"]][name][r["scheme"]].append(n90)
            data[r["maze"]][name].append(n90)
            conv[r["maze"]][name].append(r["converged"] == "True")
    return data, runs, conv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reeval", default="results/reevaluation.csv")
    p.add_argument("--runs", default="results/evolution")
    p.add_argument("--out", default="results/analysis")
    p.add_argument("--baselines", nargs="+", default=MAIN_BASELINES,
                   help="baselines to include (names or prefixes); default: the main 6")
    p.add_argument("--mazes", nargs="+", default=MAIN_MAZES, help="mazes to include")
    p.add_argument("--all", action="store_true", help="include every baseline and maze (supplementary material)")
    a = p.parse_args()
    if a.all:
        a.baselines = a.mazes = None
        a.out = a.out.rstrip("/\\") + "_all"
    os.makedirs(a.out, exist_ok=True)
    data, runs, conv = load(a.reeval, a.baselines, a.mazes)
    print(f"baselines: {'all' if a.baselines is None else ' '.join(a.baselines)}")
    print(f"mazes:     {'all' if a.mazes is None else ' '.join(a.mazes)}")

    # 1. summary table -------------------------------------------------------
    summary = []
    for maze, cands in data.items():
        print(f"\n=== {maze} ===   (median N90, lower = learns faster)")
        rows = []
        for name, v in cands.items():
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            rows.append(dict(maze=maze, candidate=name, n=len(v), median_n90=med, q1=q1, q3=q3,
                             learned=np.mean(conv[maze][name])))
        rows.sort(key=lambda r: (r["median_n90"], -r["learned"]))
        for r in rows:
            print(f"  {r['candidate']:28s} {r['median_n90']:6.0f}  [IQR {r['q1']:.0f}-{r['q3']:.0f}]"
                  f"  learned {r['learned']:4.0%}  (n={r['n']})")
        summary += rows
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # 2. statistical tests ------------------------------------------------------
    tests = []
    for maze, cands in data.items():
        baselines = [n for n in cands if n not in ("ME best", "DE best")]
        if "ME best" in runs[maze] and "DE best" in runs[maze]:
            me = [np.mean(v) for v in runs[maze]["ME best"].values()]
            de = [np.mean(v) for v in runs[maze]["DE best"].values()]
            tests.append((maze, "ME best", "DE best", "per run", me, de))
        for algo in ("ME best", "DE best"):
            if algo in cands:
                for b in baselines:
                    tests.append((maze, algo, b, "per seed", cands[algo], cands[b]))
    if not tests:
        print("\n!! No evolved candidates (ME best / DE best) in", a.reeval)
        print("!! reevaluate.py found no run files: run run_evolution.py first, then reevaluate.py again.")
        print("!! Only the baseline table above was produced.")
        return
    pvals = []
    for t in tests:
        x, y = t[4], t[5]
        pvals.append(1.0 if np.all(np.asarray(x) == x[0]) and np.all(np.asarray(y) == x[0])
                     else mannwhitneyu(x, y, alternative="two-sided").pvalue)
    adj = holm(np.array(pvals))
    stats = []
    for (maze, A, B, unit, x, y), pv, pa in zip(tests, pvals, adj):
        stats.append(dict(maze=maze, A=A, B=B, unit=unit, n_A=len(x), n_B=len(y),
                          median_A=float(np.median(x)), median_B=float(np.median(y)),
                          p=pv, p_holm=pa, A12=a12(x, y), significant=pa < ALPHA))
    with open(os.path.join(a.out, "stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    print(f"\n=== Mann-Whitney U, Holm-corrected over {len(stats)} tests (A12 > 0.5: A learns faster) ===")
    print("  shown: ME vs DE, and each algorithm vs the BEST baseline of that maze; all tests in stats.csv")
    for maze, cands in data.items():
        baselines = [n for n in cands if n not in ("ME best", "DE best")]
        best_b = min(baselines, key=lambda n: (np.median(cands[n]), np.mean(cands[n])))
        for s in stats:
            if s["maze"] == maze and (s["B"] == "DE best" or s["B"] == best_b):
                mark = "*" if s["significant"] else " "
                print(f" {mark} {maze:16s} {s['A']} vs {s['B']:28s} median {s['median_A']:.0f} vs "
                      f"{s['median_B']:.0f}  p_holm={s['p_holm']:.3g}  A12={s['A12']:.2f}  ({s['unit']})")
    print("  * = significant at p_holm < 0.05")

    # 3. boxplots ---------------------------------------------------------------
    for maze, cands in data.items():
        names = sorted(cands, key=lambda n: np.median(cands[n]))
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.2))
        bp = ax.boxplot([cands[n] for n in names], vert=False, widths=0.6, patch_artist=True,
                        medianprops=dict(color="k"))
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor("#2a9d8f" if n == "ME best" else "#e9c46a" if n == "DE best" else "#d9d9d9")
        ax.set_yticks(range(1, len(names) + 1), names, fontsize=8)
        ax.set_xlabel("episodes to 90% success (lower = better; 1000 = never learned)")
        ax.set_title(f"{maze}: fresh seeds", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"boxplot_{maze}.png"), dpi=150)
        plt.close(fig)

    # 4. convergence curves (from the evolution runs) -----------------------------
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for algo, label, col in (("mapelites", "MAP-Elites", "#2a9d8f"), ("de", "Differential Evolution", "#e9c46a")):
        curves = []
        for path in sorted(glob.glob(os.path.join(a.runs, f"{algo}_run*.json"))):
            with open(path) as f:
                h = json.load(f)["history"]
            curves.append([b for _, b, _ in h])
        if not curves:
            continue
        L = min(map(len, curves))
        c = np.array([cv[:L] for cv in curves])
        x = np.arange(1, L + 1)
        ax.plot(x, np.median(c, axis=0), color=col, label=f"{label} (median of {len(c)} runs)")
        ax.fill_between(x, np.percentile(c, 25, axis=0), np.percentile(c, 75, axis=0), color=col, alpha=0.25)
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("best fitness so far (mean N90, 5 seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "convergence.png"), dpi=150)
    print(f"\nsaved tables and figures in {a.out}/")


if __name__ == "__main__":
    main()
