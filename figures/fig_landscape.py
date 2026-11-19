"""Figure: the fitness landscape of reward space, with the evolved winners on top.

The search space has 5 genes, so it cannot be drawn whole. This figure takes a 2-D slice:
two genes vary on a grid (default: progress and step), the other genes are fixed (default:
the median of all evolved winners for trap and wall, delay 0). Every grid point is a reward
scheme; an agent is trained on new seeds and the colour shows how many episodes it needs to
learn (mean over seeds, 1000 = never learned).

On top:
    - dashed box: the range the evolution searched (evolve.BOUNDS),
    - circles / squares: the MAP-Elites / DE winners of every run, placed by their values of
      the two genes. Their other genes differ from the fixed ones, so this is a projection.

The grid goes beyond the searched range, so the figure also shows whether the range was
wide enough (see variants/).

Run from the repository folder (5-10 minutes on a laptop with the default grid):
    python figures/fig_landscape.py
    python figures/fig_landscape.py --x trap --y wall            # another slice
    python figures/fig_landscape.py --nx 13 --ny 11 --seeds 5    # quick, coarser

Output: results/figures/landscape_<x>_<y>.png / .pdf / .csv
"""
import argparse
import csv
import json
import glob
import multiprocessing as mp
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evolve  # noqa: E402
from gridmaze import get_maze  # noqa: E402
from qlearning import train  # noqa: E402
from rewards import RewardScheme  # noqa: E402

SEEDS_START = 3000            # not used by evolution (0-4), re-evaluation (1000+) or other figures (2000+)
# plotted range of each gene: about 2-3x the searched range, on the side away from zero
PLOT_RANGE = {"progress": (0.0, 3.0), "step": (-2.0, 0.0), "trap": (-40.0, 0.0), "wall": (-10.0, 0.0)}
LABEL = {"progress": "progress reward weight", "step": "step reward (cost per step)",
         "trap": "trap reward", "wall": "wall reward"}

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
ME_COLOUR, DE_COLOUR = "#2a78d6", "#eb6834"
# one neutral hue, light -> dark (dark = slow); keeps the coloured markers readable
RAMP = ["#e6e5e0", "#d6d4ce", "#c3c1b9", "#adaba2", "#96948a", "#7e7c72", "#67655c", "#504f48", "#3a3a35"]


def _job(args):
    maze_name, genes, seeds = args
    maze = get_maze(maze_name)
    scheme = RewardScheme("grid", **genes)
    n90 = []
    for s in seeds:
        r = train(maze, scheme, s)
        n90.append(r["episodes_to_90"] if r["episodes_to_90"] is not None else evolve.PENALTY)
    return dict(**genes, mean_n90=sum(n90) / len(n90), median_n90=statistics.median(n90),
                share_learned=sum(x < evolve.PENALTY for x in n90) / len(n90))


def load_winners(folder):
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "evolution", "*.json"))):
        with open(path) as f:
            run = json.load(f)
        out.append((run["algo"], evolve.decode(run["best"]["genome"])))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x", default="progress", choices=list(PLOT_RANGE))
    p.add_argument("--y", default="step", choices=list(PLOT_RANGE))
    p.add_argument("--nx", type=int, default=25, help="grid points along x")
    p.add_argument("--ny", type=int, default=21, help="grid points along y")
    p.add_argument("--seeds", type=int, default=10, help="new training seeds per grid point")
    p.add_argument("--results", default=os.path.join(ROOT, "results", "2026-11-15_laptop"),
                   help="results folder whose winners are drawn (and give the fixed gene values)")
    p.add_argument("--maze", default="dyna_9x6_3traps")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default=os.path.join(ROOT, "results", "figures"))
    p.add_argument("--from-csv", action="store_true",
                   help="redraw from the saved landscape_<x>_<y>.csv instead of training again")
    a = p.parse_args()
    if a.x == a.y:
        raise SystemExit("--x and --y must be different genes")

    winners = load_winners(a.results)
    fixed = dict(goal=evolve.GOAL, delay=0)
    for g in ("trap", "wall", "step", "progress"):
        if g not in (a.x, a.y):
            fixed[g] = round(statistics.median(getattr(s, g) for _, s in winners), 2)

    def axis(g, n):
        lo, hi = PLOT_RANGE[g]
        return [round(lo + i * (hi - lo) / (n - 1), 4) for i in range(n)]

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"landscape_{a.x}_{a.y}")
    if a.from_csv:
        with open(stem + ".csv") as f:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        xs, ys = sorted({r[a.x] for r in rows}), sorted({r[a.y] for r in rows})
        rows.sort(key=lambda r: (ys.index(r[a.y]), xs.index(r[a.x])))
        fixed = {g: rows[0][g] for g in ("goal", "delay", "trap", "wall", "step", "progress") if g not in (a.x, a.y)}
        print(f"redrawing from {stem}.csv ({len(rows)} grid points)")
    else:
        xs, ys = axis(a.x, a.nx), axis(a.y, a.ny)
        seeds = tuple(range(SEEDS_START, SEEDS_START + a.seeds))
        jobs = [(a.maze, {**fixed, a.x: x, a.y: y}, seeds) for y in ys for x in xs]
        print(f"slice {a.x} x {a.y}, fixed {fixed}; {len(jobs)} grid points x {a.seeds} seeds "
              f"on {a.workers} workers ...")
        with mp.Pool(a.workers) as pool:
            rows = pool.map(_job, jobs, chunksize=4)
        with open(stem + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    Z = np.array([r["mean_n90"] for r in rows]).reshape(len(ys), len(xs))
    best = min(rows, key=lambda r: r["mean_n90"])
    edges = [80, 100, 120, 150, 200, 300, 450, 700, 900, 1000.1]
    cmap, norm = ListedColormap(RAMP), BoundaryNorm(edges, len(RAMP))

    def cell_edges(v):
        d = (v[1] - v[0]) / 2
        return [v[0] - d] + [x + d for x in v]

    fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    mesh = ax.pcolormesh(cell_edges(xs), cell_edges(ys), Z, cmap=cmap, norm=norm, shading="flat",
                         edgecolors=SURFACE, linewidth=0.3)
    lo_x, hi_x = evolve.BOUNDS[a.x]
    lo_y, hi_y = evolve.BOUNDS[a.y]
    ax.add_patch(plt.Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y, fill=False, edgecolor=INK,
                               linewidth=1.4, linestyle=(0, (4, 3)), zorder=3))
    ax.text(lo_x + 0.02 * (hi_x - lo_x), min(lo_y, hi_y), "range searched", color=INK, fontsize=8.5,
            va="center", ha="left", zorder=3,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    for algo, colour, marker, label in (("mapelites", ME_COLOUR, "o", "MAP-Elites winners"),
                                        ("de", DE_COLOUR, "s", "DE winners")):
        pts = [(getattr(s, a.x), getattr(s, a.y)) for al, s in winners if al == algo]
        if pts:
            ax.scatter(*zip(*pts), s=46, marker=marker, color=colour, edgecolor=SURFACE, linewidth=1.2,
                       zorder=4, label=f"{label} ({len(pts)} runs)")
    ax.scatter([best[a.x]], [best[a.y]], s=90, marker="*", color=INK, edgecolor=SURFACE, linewidth=0.8,
               zorder=5, label=f"fastest grid point ({best['mean_n90']:.0f} episodes)")

    ax.set_xlim(cell_edges(xs)[0], cell_edges(xs)[-1])
    ax.set_ylim(cell_edges(ys)[0], cell_edges(ys)[-1])
    ax.set_xlabel(LABEL[a.x], color=MUTED, fontsize=10)
    ax.set_ylabel(LABEL[a.y], color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02, ticks=edges[:-1] + [1000])
    cb.set_label("episodes to learn (mean of seeds; 1000 = never)", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    cb.outline.set_visible(False)
    leg = ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="none",
                    facecolor=SURFACE)
    for t in leg.get_texts():
        t.set_color(INK)
    fixed_txt = ", ".join(f"{g} {v:g}" for g, v in fixed.items() if g != "goal")
    fig.text(0.01, 0.985, f"Fitness landscape: {a.x} x {a.y} ({a.maze})", fontsize=12, color=INK, va="top")
    fig.text(0.01, 0.94, f"Each cell is one reward scheme: goal +10, {fixed_txt} (median of the winners), "
             f"{a.x} and {a.y} as on the axes.\nDarker = slower, mean of {a.seeds} new seeds. The winners' other "
             f"genes differ from the fixed ones, so they are projected onto this slice.",
             fontsize=8, color=MUTED, va="top", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(stem + ".png", dpi=170, facecolor=SURFACE)
    fig.savefig(stem + ".pdf", facecolor=SURFACE)

    inside = [r for r in rows if lo_x <= r[a.x] <= hi_x and min(lo_y, hi_y) <= r[a.y] <= max(lo_y, hi_y)]
    best_in = min(inside, key=lambda r: r["mean_n90"])
    print(f"fastest point overall:            {a.x} {best[a.x]:g}, {a.y} {best[a.y]:g}: {best['mean_n90']:.0f} episodes")
    print(f"fastest point inside the range:   {a.x} {best_in[a.x]:g}, {a.y} {best_in[a.y]:g}: "
          f"{best_in['mean_n90']:.0f} episodes")
    print(f"saved {stem}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
