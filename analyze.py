"""Compare MAP-Elites, Differential Evolution and the hand-made baselines.

Reads the outputs of run_evolution.py and reevaluate.py and produces (in <dir>/analysis/):

  1. summary.csv + printed table
       per maze and candidate: median episodes to 90% (N90), IQR, share of seeds that learned
  2. stats.csv, stats_summary.csv + printed table
       Mann-Whitney U tests with Holm correction and Vargha-Delaney A12 effect size for
         ME vs DE                 (unit = one evolutionary run: mean N90 over the fresh seeds)
         every evolved scheme vs every baseline, one test per scheme (30 vs 30 fresh seeds)
       The winners of different runs are DIFFERENT reward schemes, so they are tested one by one
       and not pooled; stats_summary.csv counts how many of them are significantly faster / slower.
  3. boxplot_<maze>.png      N90 of every candidate on that maze
  4. convergence.png         best fitness vs evaluations, ME vs DE (median + IQR)
  5. qd.csv + printed table  quality-diversity of what each algorithm returns, on the MAP-Elites grid:
       cells filled, cells whose elite learned on every evaluation seed, QD-score
       (MAP-Elites: its archive; DE: its final population placed on the same grid)

Examples:
    python reevaluate.py --runs results/evolution --seeds 30       # first
    python analyze.py                                              # main 6 baselines, 5 mazes -> results/analysis/
    python analyze.py --dir results/2026-11-15_laptop              # a saved snapshot of results
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

import evolve

ALPHA = 0.05
EVOLVED = ("ME best", "DE best")
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


def mw_p(x, y):
    """Two-sided Mann-Whitney U p-value (1.0 when every value in both samples is identical)."""
    x, y = np.asarray(x), np.asarray(y)
    if np.all(x == x[0]) and np.all(y == x[0]):
        return 1.0
    return mannwhitneyu(x, y, alternative="two-sided").pvalue


def qd_metrics(run):
    """Quality-diversity of what one evolutionary run returns, on the MAP-Elites grid.

    MAP-Elites returns its archive. DE returns its final population, which is placed on the same
    grid here (best individual per cell); run files do not store every genome DE evaluated.
    QD-score = sum over filled cells of (PENALTY - fitness): an empty cell and a cell whose elite
    never learns both add 0, a faster elite adds more.
    """
    if run["algo"] == "mapelites":
        cells = {tuple(e["cell"]): e for e in run["archive"]}
    else:
        cells = {}
        for e in run["population"] + [run["best"]]:
            c = evolve.cell_of(e)
            if c not in cells or e["fitness"] < cells[c]["fitness"]:
                cells[c] = e
    return dict(algo=run["algo"], run=run["run"],
                cells_filled=len(cells),
                cells_learning=sum(e["converged"] == len(e["n90"]) for e in cells.values()),
                qd_score=sum(evolve.PENALTY - e["fitness"] for e in cells.values()),
                best_fitness=min(e["fitness"] for e in cells.values()))


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
    p.add_argument("--dir", default="results",
                   help="results folder holding reevaluation.csv and evolution/ (default: results)")
    p.add_argument("--reeval", help="default: <dir>/reevaluation.csv")
    p.add_argument("--runs", help="default: <dir>/evolution")
    p.add_argument("--out", help="default: <dir>/analysis")
    p.add_argument("--baselines", nargs="+", default=MAIN_BASELINES,
                   help="baselines to include (names or prefixes); default: the main 6")
    p.add_argument("--mazes", nargs="+", default=MAIN_MAZES, help="mazes to include")
    p.add_argument("--all", action="store_true", help="include every baseline and maze (supplementary material)")
    a = p.parse_args()
    a.reeval = a.reeval or os.path.join(a.dir, "reevaluation.csv")
    a.runs = a.runs or os.path.join(a.dir, "evolution")
    a.out = a.out or os.path.join(a.dir, "analysis")
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
    # ME vs DE: one value per evolutionary run. Evolved vs baseline: one test per evolved scheme.
    tests = []                       # (maze, group, A, B, unit, x, y)
    for maze, cands in data.items():
        baselines = [n for n in cands if n not in EVOLVED]
        if all(algo in runs[maze] for algo in EVOLVED):
            me = [np.mean(v) for v in runs[maze]["ME best"].values()]
            de = [np.mean(v) for v in runs[maze]["DE best"].values()]
            tests.append((maze, "ME vs DE", "ME best", "DE best", "per run", me, de))
        for algo in EVOLVED:
            for scheme, v in runs[maze][algo].items():
                for b in baselines:
                    tests.append((maze, algo, scheme, b, "per scheme", v, cands[b]))
    if not tests:
        print("\n!! No evolved candidates (ME best / DE best) in", a.reeval)
        print("!! reevaluate.py found no run files: run run_evolution.py first, then reevaluate.py again.")
        print("!! Only the baseline table above was produced.")
        return
    pvals = [mw_p(t[5], t[6]) for t in tests]
    adj = holm(np.array(pvals))
    stats = []
    for (maze, group, A, B, unit, x, y), pv, pa in zip(tests, pvals, adj):
        eff = a12(x, y)
        verdict = "no difference" if pa >= ALPHA else "faster" if eff > 0.5 else "slower"
        stats.append(dict(maze=maze, group=group, A=A, B=B, unit=unit, n_A=len(x), n_B=len(y),
                          median_A=float(np.median(x)), median_B=float(np.median(y)),
                          p=pv, p_holm=pa, A12=eff, significant=pa < ALPHA, verdict=verdict))
    with open(os.path.join(a.out, "stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)

    # how many evolved schemes of each algorithm beat each baseline
    agg = []
    for maze, cands in data.items():
        baselines = sorted((n for n in cands if n not in EVOLVED),
                           key=lambda n: (np.median(cands[n]), np.mean(cands[n])))
        for b in baselines:
            for algo in EVOLVED:
                rows = [s for s in stats if s["maze"] == maze and s["group"] == algo and s["B"] == b]
                if rows:
                    agg.append(dict(maze=maze, algo=algo, baseline=b, schemes=len(rows),
                                    faster=sum(s["verdict"] == "faster" for s in rows),
                                    slower=sum(s["verdict"] == "slower" for s in rows),
                                    no_difference=sum(s["verdict"] == "no difference" for s in rows),
                                    median_A12=float(np.median([s["A12"] for s in rows])),
                                    median_n90_evolved=float(np.median(cands[algo])),
                                    median_n90_baseline=float(np.median(cands[b]))))
    with open(os.path.join(a.out, "stats_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)

    print(f"\n=== Mann-Whitney U, two-sided, Holm-corrected over all {len(stats)} tests ===")
    print("  ME vs DE: one value per evolutionary run (mean N90 over the fresh seeds); A12 > 0.5 = ME faster")
    print("  evolved vs baseline: every evolved scheme tested on its own (fresh seeds vs fresh seeds);")
    print("  shown: how many schemes are significantly faster / slower than the baseline, and their median A12")
    for maze, cands in data.items():
        print(f"\n  {maze}")
        for s in stats:
            if s["maze"] == maze and s["unit"] == "per run":
                mark = "*" if s["significant"] else " "
                print(f"  {mark} ME vs DE (per-run mean N90)     median {s['median_A']:.0f} vs {s['median_B']:.0f}"
                      f"  p_holm={s['p_holm']:.3g}  A12={s['A12']:.2f}  (n={s['n_A']}/{s['n_B']} runs)")
        for b in dict.fromkeys(g["baseline"] for g in agg if g["maze"] == maze):
            parts = []
            for g in agg:
                if g["maze"] == maze and g["baseline"] == b:
                    parts.append(f"{g['algo'][:2]}: {g['faster']:2d}/{g['schemes']} faster, {g['slower']}/{g['schemes']} slower"
                                 f" (A12 {g['median_A12']:.2f})")
            print(f"    vs {b:27s} [{np.median(cands[b]):4.0f}]  " + "   ".join(parts))
    print("\n  * = significant at p_holm < 0.05; [..] = median N90 of the baseline; all tests in stats.csv")

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
    plt.close(fig)

    # 5. quality-diversity of what each algorithm returns ---------------------------
    qd = []
    for path in sorted(glob.glob(os.path.join(a.runs, "*_run*.json"))):
        with open(path) as f:
            qd.append(qd_metrics(json.load(f)))
    if qd:
        with open(os.path.join(a.out, "qd.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(qd[0]))
            w.writeheader()
            w.writerows(qd)
        n_cells = evolve.GRID[0] * evolve.GRID[1]
        print(f"\n=== Quality-diversity on the {evolve.GRID[0]}x{evolve.GRID[1]} MAP-Elites grid ({n_cells} cells) ===")
        print("  MAP-Elites: its archive.  DE: its final population placed on the same grid.")
        print("  median over runs [min-max];  cells learning = elite reached 90% on every evaluation seed;")
        print(f"  QD-score = sum over filled cells of ({evolve.PENALTY} - fitness)")
        by = {algo: [q for q in qd if q["algo"] == algo] for algo in ("mapelites", "de")}
        labels = (("cells_filled", "cells filled"), ("cells_learning", "cells learning"), ("qd_score", "QD-score"))
        for key, label in labels:
            line = f"  {label:15s}"
            for algo, name in (("mapelites", "MAP-Elites"), ("de", "DE")):
                v = [q[key] for q in by[algo]]
                if v:
                    line += f"  {name} {np.median(v):8.0f} [{min(v):.0f}-{max(v):.0f}] (n={len(v)})"
            if len(by["mapelites"]) > 1 and len(by["de"]) > 1:
                line += f"   Mann-Whitney p={mw_p([q[key] for q in by['mapelites']], [q[key] for q in by['de']]):.2g}"
            print(line)
    print(f"\nsaved tables and figures in {a.out}/")


if __name__ == "__main__":
    main()
