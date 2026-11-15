"""Cheap test of the gene ranges WITHOUT running evolution again.

Take the evolved winners of a finished experiment. Change ONE reward gene at a time to a
fixed value (also far outside the range that was searched), keep the other genes as
evolved, retrain on new seeds and measure the learning speed.

How to read the result, for each gene:
    - learning gets faster outside the searched range  -> widening this gene could help
    - flat or slower outside the range                 -> the range is already wide enough
This is a one-gene-at-a-time test around the winners. It cannot see combinations of
several larger genes; only an evolutionary run with a wider variant can (run_variant.py).

Run from the repository folder (about 5 minutes on a laptop):
    python variants/sweep_genes.py
    python variants/sweep_genes.py --runs results/variants/wide_x2 --seeds 20

Output in results/variants/sweep/: sweep.csv (every training run), sweep_summary.csv, sweep.png
"""
import argparse
import csv
import dataclasses
import multiprocessing as mp
import os
import statistics
from collections import defaultdict

import variant_defs
import evolve
from gridmaze import get_maze
from qlearning import train
from rewards import RewardScheme

# Values tried for each gene. The default search range is trap -20..0, wall -5..0,
# step -1..0, progress 0..1, so every list goes to 4x the default range.
GRIDS = {
    "progress": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4],
    "step": [0, -0.25, -0.5, -0.75, -1, -1.5, -2, -3, -4],
    "trap": [0, -5, -10, -20, -40, -80],
    "wall": [0, -1.25, -2.5, -5, -10, -20],
}
SEEDS_START = 2000          # evolution used seeds 0-4, the re-evaluation 1000-1029
ALGO_LABEL = {"mapelites": "ME winners", "de": "DE winners"}


def _job(args):
    maze_name, gene, value, algo, run, scheme, seed = args
    r = train(get_maze(maze_name), scheme, seed)
    n90 = r["episodes_to_90"]
    return dict(gene=gene, value=value, algo=algo, run=run, seed=seed,
                episodes_to_90=n90 if n90 is not None else evolve.PENALTY, converged=n90 is not None)


def winner_scheme(run, bounds):
    if "best_scheme" in run:
        return RewardScheme(**run["best_scheme"])
    variant_defs.apply_bounds(bounds)
    return evolve.decode(run["best"]["genome"])


def summarise(rows):
    """One line per (gene, value, group of winners): the typical winner and the spread."""
    per_winner = defaultdict(list)
    for r in rows:
        per_winner[(r["gene"], r["value"], r["algo"], r["run"])].append(r)
    groups = defaultdict(list)
    for (gene, value, algo, run), rs in per_winner.items():
        groups[(gene, value, algo)].append(
            (statistics.median(x["episodes_to_90"] for x in rs), sum(x["converged"] for x in rs) / len(rs)))
    out = []
    for (gene, value, algo), ws in groups.items():
        med = sorted(w[0] for w in ws)
        q = statistics.quantiles(med, n=4) if len(med) > 1 else [med[0]] * 3
        out.append(dict(gene=gene, value=value, winners=ALGO_LABEL.get(algo, algo), n_winners=len(ws),
                        median_n90=statistics.median(med), q1_n90=q[0], q3_n90=q[2],
                        pct_learned=round(100 * statistics.mean(w[1] for w in ws), 1)))
    order = ["as evolved"] + list(GRIDS)
    out.sort(key=lambda r: (order.index(r["gene"]), r["winners"], abs(r["value"])))
    return out


def plot(summary, bounds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colour = {"ME winners": "#2a78d6", "DE winners": "#eb6834"}
    marker = {"ME winners": "o", "DE winners": "s"}
    genes = list(GRIDS)
    fig, axes = plt.subplots(2, len(genes), figsize=(3.4 * len(genes), 6.2), sharex="col", facecolor=surface)
    for c, gene in enumerate(genes):
        lo, hi = bounds[gene]
        for r_i, (key, ylabel) in enumerate([("median_n90", "episodes to learn (median)"),
                                             ("pct_learned", "runs that learned (%)")]):
            ax = axes[r_i][c]
            ax.set_facecolor(surface)
            ax.axvspan(lo, hi, color="#ecebe7", zorder=0, label="range searched so far" if c == 0 else None)
            for label in colour:
                pts = sorted((r["value"], r) for r in summary if r["gene"] == gene and r["winners"] == label)
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ax.plot(xs, [p[1][key] for p in pts], color=colour[label], marker=marker[label], lw=2, ms=6,
                        markeredgecolor=surface, markeredgewidth=1, label=label if c == 0 else None, zorder=3)
                if key == "median_n90":
                    ax.fill_between(xs, [p[1]["q1_n90"] for p in pts], [p[1]["q3_n90"] for p in pts],
                                    color=colour[label], alpha=0.12, lw=0, zorder=2)
            ax.set_ylim((0, evolve.PENALTY * 1.03) if key == "median_n90" else (0, 103))
            ax.grid(axis="y", color=grid, lw=0.8)
            ax.tick_params(colors=muted, labelsize=9, length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(grid)
            if c == 0:
                ax.set_ylabel(ylabel, color=muted, fontsize=10)
            if r_i == 0:
                ax.set_title(f"{gene} reward", color=ink, fontsize=11, loc="left")
            else:
                ax.set_xlabel(f"value of the {gene} gene", color=muted, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=10, labelcolor=ink)
    fig.suptitle("One gene of the evolved winners changed at a time", x=0.01, ha="left", color=ink, fontsize=13)
    fig.text(0.01, 0.925, f"Top: lower is faster; {evolve.PENALTY} = never learned. Band = middle half of the winners.",
             color=muted, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=surface)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=os.path.join(variant_defs.ROOT, "results", "2026-11-15_laptop"),
                   help="results folder with the winners (contains evolution/*.json)")
    p.add_argument("--maze", default="dyna_9x6_3traps")
    p.add_argument("--seeds", type=int, default=10, help="new training seeds per winner and value")
    p.add_argument("--out", default=os.path.join(variant_defs.ROOT, "results", "variants", "sweep"))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    runs = variant_defs.load_runs(a.runs)
    if not runs:
        raise SystemExit(f"no run files found in {a.runs}")
    bounds = variant_defs.bounds_of_runs(runs)
    winners = [(r["algo"], r["run"], winner_scheme(r, bounds)) for r in runs]
    seeds = range(SEEDS_START, SEEDS_START + a.seeds)

    jobs = [(a.maze, "as evolved", 0, algo, run, s, seed) for algo, run, s in winners for seed in seeds]
    for gene, values in GRIDS.items():
        for v in values:
            jobs += [(a.maze, gene, v, algo, run, dataclasses.replace(s, **{gene: float(v)}), seed)
                     for algo, run, s in winners for seed in seeds]
    print(f"{len(winners)} winners from {a.runs}; {len(jobs)} training runs on {a.workers} workers ...")
    with mp.Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=16)

    os.makedirs(a.out, exist_ok=True)
    for name, data in [("sweep.csv", rows), ("sweep_summary.csv", summarise(rows))]:
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    summary = summarise(rows)

    print(f"\n{'gene':10s} {'value':>6s}  " + "   ".join(f"{g:>22s}" for g in ALGO_LABEL.values()))
    print(" " * 19 + "   ".join(f"{'median N90   learned':>22s}" for _ in ALGO_LABEL))
    keys = []
    for r in summary:
        if (r["gene"], r["value"]) not in keys:
            keys.append((r["gene"], r["value"]))
    for gene, value in keys:
        cells = []
        for label in ALGO_LABEL.values():
            m = [r for r in summary if (r["gene"], r["value"], r["winners"]) == (gene, value, label)]
            cells.append(f"{m[0]['median_n90']:10.0f} {m[0]['pct_learned']:8.0f} %" if m else " " * 22)
        inside = ""
        if gene in bounds:
            lo, hi = bounds[gene]
            inside = "" if lo <= value <= hi else "   (outside the searched range)"
        print(f"{gene:10s} {value:6g}  " + "   ".join(f"{c:>22s}" for c in cells) + inside)

    try:
        plot([r for r in summary if r["gene"] != "as evolved"], bounds, os.path.join(a.out, "sweep.png"))
    except ImportError:
        print("matplotlib not installed: no figure")
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
