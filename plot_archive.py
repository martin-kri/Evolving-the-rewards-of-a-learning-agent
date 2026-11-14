"""Draw the MAP-Elites archive(s) as heatmaps.

Each cell of the 10x10 grid is one "teaching style":
    x axis = mean feedback delay (steps)    y axis = punishment share
The colour is the fitness of the elite in that cell (episodes to 90%, lower = better);
white cells were never reached. Cells where some seeds never learned are marked with a dot.

Examples:
    python plot_archive.py results/evolution/mapelites_run0.json
    python plot_archive.py results/evolution/mapelites_run*.json      # one panel per run + the median
    python plot_archive.py results/evolution/mapelites_run0.json --gene trap   # colour = a gene value
Output: next to the first file, e.g. results/evolution/archive_mapelites_run0.png
"""
import argparse
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evolve


def load_grid(path, gene=None):
    with open(path) as f:
        run = json.load(f)
    grid = np.full(evolve.GRID, np.nan)
    unconverged = np.zeros(evolve.GRID, dtype=bool)
    for e in run["archive"]:
        i, j = e["cell"]
        if gene:
            grid[i, j] = getattr(evolve.decode(e["genome"]), gene)
        else:
            grid[i, j] = e["fitness"]
        unconverged[i, j] = e["converged"] < len(e["n90"])
    return run, grid, unconverged


def draw(ax, grid, unconverged, title, vmin, vmax, cmap):
    im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, evolve.MAX_DELAY, 0, 1], aspect="auto")
    di, dj = 1 / evolve.GRID[0], evolve.MAX_DELAY / evolve.GRID[1]
    ii, jj = np.nonzero(unconverged)
    ax.scatter((jj + 0.5) * dj, (ii + 0.5) * di, s=6, c="k")
    ax.set_xlabel("mean feedback delay (steps)")
    ax.set_ylabel("punishment share")
    ax.set_title(title, fontsize=9)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--gene", choices=evolve.GENES, help="colour by this gene instead of fitness")
    p.add_argument("--out")
    a = p.parse_args()

    loaded = [load_grid(f, a.gene) for f in a.files]
    grids = [g for _, g, _ in loaded]
    panels = [(f"run {r['run']}  (cells: {np.sum(~np.isnan(g))}, best: {np.nanmin(g):.0f})"
               if not a.gene else f"run {r['run']}", g, u) for r, g, u in loaded]
    if len(grids) > 1:
        with np.errstate(all="ignore"):
            import warnings
            warnings.simplefilter("ignore")
            med = np.nanmedian(np.stack(grids), axis=0)
        panels.append((f"median over {len(grids)} runs", med, np.zeros_like(med, dtype=bool)))

    all_vals = np.concatenate([g[~np.isnan(g)] for _, g, _ in panels])
    vmin, vmax = (np.min(all_vals), np.percentile(all_vals, 95)) if not a.gene else (all_vals.min(), all_vals.max())
    cmap = "viridis_r" if not a.gene else "coolwarm"
    n = len(panels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    for ax, (title, g, u) in zip(axes.flat, panels):
        im = draw(ax, g, u, title, vmin, vmax, cmap)
    for ax in axes.flat[n:]:
        ax.axis("off")
    label = a.gene if a.gene else "fitness: mean episodes to 90% (lower = better)"
    fig.colorbar(im, ax=axes, shrink=0.85, label=label)
    out = a.out or os.path.join(os.path.dirname(a.files[0]),
                                f"archive_{a.gene + '_' if a.gene else ''}{os.path.basename(a.files[0])[:-5]}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
