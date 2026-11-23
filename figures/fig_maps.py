"""Figure: MAP-Elites map next to DE map, and how many cells each covers over time.

Panels (same grid as the MAP-Elites archive: punishment share x mean feedback delay):
    (a) MAP-Elites: its archive at the end of one run, coloured by fitness.
    (b) DE: every cell that DE EVALUATED at some point in the same run (best scheme per cell),
        and the cell(s) of its final population (orange outline). DE keeps no map; this is
        the most generous way to place it on one. Needs the trace from trace_run.py; without
        it only the final population is drawn.
    (c) cells covered vs fitness evaluations: MAP-Elites archive size (all runs: median line,
        band = min to max), the random-search map if there are random_run*.json files (same),
        and cells visited by DE (every traced DE run).

Run from the repository folder (seconds):
    python figures/fig_maps.py
    python figures/fig_maps.py --run 3 --trace-dir results/figures/trace

Output: results/figures/maps_run<N>.png / .pdf
"""
import argparse
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evolve  # noqa: E402

INK, MUTED, GRIDLINE, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EMPTY = SURFACE                     # empty cells: white with a thin outline
ME_COLOUR, DE_COLOUR, RS_COLOUR = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#e6e5e0", "#d6d4ce", "#c3c1b9", "#adaba2", "#96948a", "#7e7c72", "#67655c", "#504f48", "#3a3a35"]
EDGES = [80, 100, 120, 150, 200, 300, 450, 700, 900, 1000.1]   # same colour bins as fig_landscape.py


def load(path):
    with open(path) as f:
        return json.load(f)


def best_per_cell(evs):
    cells = {}
    for ev in evs:
        c = evolve.cell_of(ev)
        if c not in cells or ev["fitness"] < cells[c]["fitness"]:
            cells[c] = ev
    return cells


def coverage_curve(trace):
    """Number of distinct cells visited after each evaluation."""
    seen, curve = set(), []
    for ev in trace:
        seen.add(evolve.cell_of(ev))
        curve.append(len(seen))
    return curve


def draw_map(ax, cells, title, outline=(), cmap=None, norm=None):
    import matplotlib.patches as mpatches
    n_i, n_j = evolve.GRID
    for i in range(n_i):
        for j in range(n_j):
            ev = cells.get((i, j))
            fc = cmap(norm(ev["fitness"])) if ev else EMPTY
            ax.add_patch(mpatches.Rectangle((j, i), 1, 1, facecolor=fc, edgecolor=SURFACE, linewidth=1.2))
            if not ev:
                ax.add_patch(mpatches.Rectangle((j + 0.08, i + 0.08), 0.84, 0.84, fill=False,
                                                edgecolor=GRIDLINE, linewidth=0.7))
            if ev and ev["converged"] < len(ev["n90"]):
                ax.plot(j + 0.5, i + 0.5, marker="o", ms=2.2, color=INK if ev["fitness"] < 450 else SURFACE)
    for (i, j) in outline:
        ax.add_patch(mpatches.Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, fill=False, edgecolor=DE_COLOUR,
                                        linewidth=2.2, zorder=4))
    ax.set_xlim(0, n_j)
    ax.set_ylim(0, n_i)
    ax.set_aspect("equal")
    ax.set_xticks([0, n_j / 2, n_j])
    ax.set_xticklabels(["0", f"{evolve.MAX_DELAY // 2}", f"{evolve.MAX_DELAY}"])
    ax.set_yticks([0, n_i / 2, n_i])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("mean feedback delay (steps)", color=MUTED, fontsize=9)
    ax.set_ylabel("punishment share", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, color=INK)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(ROOT, "results", "2026-11-15_laptop"))
    p.add_argument("--trace-dir", default=os.path.join(ROOT, "results", "figures", "trace"))
    p.add_argument("--run", type=int, default=7, help="run shown in the two maps")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "figures"))
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap, norm = ListedColormap(RAMP), BoundaryNorm(EDGES, len(RAMP))
    me = load(os.path.join(a.results, "evolution", f"mapelites_run{a.run}.json"))
    me_cells = {tuple(e["cell"]): e for e in me["archive"]}
    de_trace_path = os.path.join(a.trace_dir, f"de_run{a.run}.json")
    if os.path.exists(de_trace_path):
        de = load(de_trace_path)
        de_cells = best_per_cell(de["trace"])
        last_new = max(i for i, n in enumerate(coverage_curve(de["trace"])) if i == 0 or
                       n > coverage_curve(de["trace"])[i - 1]) + 1
        de_title = f"(b) DE, run {a.run}: cells it evaluated ({len(de_cells)})"
    else:
        print(f"no trace for DE run {a.run} in {a.trace_dir}: drawing the final population only")
        de = load(os.path.join(a.results, "evolution", f"de_run{a.run}.json"))
        de_cells, last_new = best_per_cell(de["population"]), None
        de_title = f"(b) DE, run {a.run}: final population"
    final_cells = {evolve.cell_of(e) for e in de["population"]}

    fig = plt.figure(figsize=(11.6, 4.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.055, 0.12, 1.25], wspace=0.30,
                          left=0.05, right=0.985, top=0.80, bottom=0.20)
    ax_me, ax_de, cax, ax_cov = (fig.add_subplot(gs[0, k]) for k in (0, 1, 2, 4))
    for ax in (ax_me, ax_de, ax_cov):
        ax.set_facecolor(SURFACE)
    learning = sum(e["converged"] == len(e["n90"]) for e in me_cells.values())
    draw_map(ax_me, me_cells, f"(a) MAP-Elites, run {a.run}: archive ({len(me_cells)} cells)", cmap=cmap, norm=norm)
    draw_map(ax_de, de_cells, de_title, outline=final_cells, cmap=cmap, norm=norm)
    ax_me.text(0, -0.24, f"{learning} of {len(me_cells)} elites learn on every seed", transform=ax_me.transAxes,
               fontsize=8, color=MUTED)
    ax_de.text(0, -0.24, f"orange outline: final population ({len(final_cells)} cell"
               f"{'s' if len(final_cells) > 1 else ''})", transform=ax_de.transAxes, fontsize=8, color=MUTED)

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=EDGES[:-1] + [1000])
    cb.set_label("episodes to learn (fitness)", color=MUTED, fontsize=8.5)
    cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    cb.outline.set_visible(False)

    # (c) coverage over evaluations
    me_runs = [load(f) for f in sorted(glob.glob(os.path.join(a.results, "evolution", "mapelites_run*.json")))]
    curves = [[h[2] for h in r["history"]] for r in me_runs]
    n = min(len(c) for c in curves)
    xs = list(range(1, n + 1))
    med = [statistics.median(c[k] for c in curves) for k in range(n)]
    lo = [min(c[k] for c in curves) for k in range(n)]
    hi = [max(c[k] for c in curves) for k in range(n)]
    ax_cov.fill_between(xs, lo, hi, color=ME_COLOUR, alpha=0.15, lw=0)
    ax_cov.plot(xs, med, color=ME_COLOUR, lw=2, label=f"MAP-Elites archive ({len(curves)} run{'s' if len(curves) > 1 else ''})")
    ax_cov.text(xs[-1], med[-1] + 2.5, f"{med[-1]:.0f}", color=INK, fontsize=8.5, ha="right", va="bottom")
    rs_runs = [load(f) for f in sorted(glob.glob(os.path.join(a.results, "evolution", "random_run*.json")))]
    if rs_runs:                       # random search: same map and insertion rule, no evolution
        rc = [[h[2] for h in r["history"]] for r in rs_runs]
        m = min(len(c) for c in rc)
        rmed = [statistics.median(c[k] for c in rc) for k in range(m)]
        ax_cov.fill_between(range(1, m + 1), [min(c[k] for c in rc) for k in range(m)],
                            [max(c[k] for c in rc) for k in range(m)], color=RS_COLOUR, alpha=0.15, lw=0)
        ax_cov.plot(range(1, m + 1), rmed, color=RS_COLOUR, lw=2, label=f"random-search map ({len(rc)} run{'s' if len(rc) > 1 else ''})")
        ax_cov.text(m, rmed[-1] - 2.5, f"{rmed[-1]:.0f}", color=INK, fontsize=8.5, ha="right", va="top")
    de_traces = [load(f) for f in sorted(glob.glob(os.path.join(a.trace_dir, "de_run*.json")))]
    for k, d in enumerate(de_traces):
        c = coverage_curve(d["trace"])
        ax_cov.plot(range(1, len(c) + 1), c, color=DE_COLOUR, lw=2 if k == 0 else 1.2, alpha=1 if k == 0 else 0.6,
                    label=f"cells visited by DE ({len(de_traces)} traced run{'s' if len(de_traces) > 1 else ''})"
                    if k == 0 else None)
    if de_traces:
        c = coverage_curve(de_traces[0]["trace"])
        ax_cov.text(len(c), c[-1] + 2.5, f"{c[-1]}", color=INK, fontsize=8.5, ha="right", va="bottom")
    if last_new:
        ax_cov.annotate(f"DE finds its last new cell\nat evaluation {last_new}", xy=(last_new, len(de_cells)),
                        xytext=(last_new * 1.6, len(de_cells) - 19), fontsize=8, color=INK, ha="left",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax_cov.set_xscale("log")
    ax_cov.set_xlim(1, n)
    ax_cov.set_ylim(0, evolve.GRID[0] * evolve.GRID[1] * 0.75)
    ax_cov.set_xticks([1, 10, 100, 1000, n])
    ax_cov.set_xticklabels(["1", "10", "100", "1000", f"{n}"])
    ax_cov.set_xlabel("fitness evaluations (log scale)", color=MUTED, fontsize=9)
    ax_cov.set_ylabel("map cells covered (of 100)", color=MUTED, fontsize=9)
    ax_cov.grid(axis="y", color=GRIDLINE, lw=0.8)
    ax_cov.tick_params(colors=MUTED, labelsize=8, length=0)
    for s in ("top", "right", "left"):
        ax_cov.spines[s].set_visible(False)
    ax_cov.spines["bottom"].set_color(GRIDLINE)
    ax_cov.set_title("(c) Map cells covered over the run", loc="left", fontsize=10, color=INK)
    leg = ax_cov.legend(loc="upper left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.text(0.01, 0.975, "Where in the map each algorithm puts its reward schemes (training maze, budget "
             f"{me['budget']} evaluations)", fontsize=12, color=INK, va="top")
    fig.text(0.01, 0.915, "Map axes: share of the feedback that is punishment, and average delay of the "
             "feedback. Darker = slower to learn; dot = did not learn on every evaluation seed; "
             "white = empty.", fontsize=8, color=MUTED, va="top")
    # the maps keep a square shape; give the colour bar and panel (c) the same height
    fig.canvas.draw()
    m = ax_me.get_position()
    for ax in (cax, ax_cov):
        q = ax.get_position()
        ax.set_position([q.x0, m.y0, q.width, m.height])

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"maps_run{a.run}")
    fig.savefig(stem + ".png", dpi=170, facecolor=SURFACE)
    fig.savefig(stem + ".pdf", facecolor=SURFACE)
    print(f"MAP-Elites run {a.run}: {len(me_cells)} cells ({learning} learning); DE run {a.run}: "
          f"{len(de_cells)} cells evaluated, final population in {len(final_cells)}"
          + (f", last new cell at evaluation {last_new}" if last_new else ""))
    print(f"saved {stem}.png / .pdf")


if __name__ == "__main__":
    main()
