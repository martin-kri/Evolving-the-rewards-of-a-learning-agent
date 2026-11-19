"""Figure: what the agent learns under different reward schemes, drawn on the maze.

For each reward scheme, train one agent per seed (new seeds, not used anywhere else) and draw:
    - lines: the route of the final greedy policy (no slip) from S; line width = number of
      seeds whose agent took that step, so a thick line = the usual route,
    - with --shading, also where the agents walked during the first 100 training episodes
      (the learning phase), averaged over the seeds; darker = visited more. Off by default:
      on the training maze the shading hardly differs between schemes.
Under each panel: median episodes to learn, and how many seeds took the upper route, the
lower route, or never reach the goal with the final greedy policy.

Default panels: three baselines, the DE and MAP-Elites winners of one run, and the best
MAP-Elites elite of a chosen map cell (an equally fast scheme with different behaviour).

Run from the repository folder (about a minute):
    python figures/fig_maze_paths.py
    python figures/fig_maze_paths.py --run 3 --elite-cell 4 0 --seeds 30
    python figures/fig_maze_paths.py --maze maze_8x8

Output: results/figures/maze_paths_<maze>.png and .pdf (vector, for the report), and
maze_paths_<maze>.csv with the numbers.

Implementation note: qlearning.train() does not return the Q-table, so this script wraps
qlearning.evaluate() (which train() calls with the Q-table) and the maze's step() function
to read the Q-table and count visits. qlearning.py itself is not changed.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evolve  # noqa: E402
import qlearning  # noqa: E402
from gridmaze import Maze, get_maze  # noqa: E402
from rewards import baseline_schemes  # noqa: E402

SEEDS_START = 2000           # evolution used 0-4, re-evaluation 1000-1029
EARLY_EPISODES = 100         # "learning phase" shown by the cell shading

INK, MUTED, FAINT, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
WALL, TRAP_FILL, TRAP_MARK = "#3d3d3a", "#f6d3d2", "#b23b3a"
GROUP_COLOUR = {"baseline": "#52514e", "DE": "#eb6834", "ME": "#2a78d6"}
SHADE = ["#fcfcfb", "#f0efec", "#e2e1dc", "#cfcdc6", "#b8b6ae", "#9e9c93"]   # one neutral hue, light -> dark


# ---------------------------------------------------------------- reading the agent
class _Spy:
    """Collects the Q-table and the visits of one training run."""
    def __init__(self):
        self.Q, self.tests, self.in_test, self.visits = None, 0, False, Counter()


_spy = _Spy()
_original_evaluate = qlearning.evaluate


def _evaluate_and_capture(maze, Q, *args, **kwargs):
    _spy.Q = Q
    _spy.tests += 1
    _spy.in_test = True
    try:
        return _original_evaluate(maze, Q, *args, **kwargs)
    finally:
        _spy.in_test = False


qlearning.evaluate = _evaluate_and_capture       # train() looks this name up at call time


def train_and_read(maze, scheme, seed):
    """Train one agent. Returns (result dict, final Q-table, visits in the first EARLY_EPISODES)."""
    global _spy
    _spy = _Spy()
    every = qlearning.DEFAULTS["eval_every"]

    def counting_step(s, a, rng):
        out = Maze.step(maze, s, a, rng)
        if not _spy.in_test and _spy.tests * every < EARLY_EPISODES:
            _spy.visits[out[0]] += 1
        return out

    maze.step = counting_step
    try:
        r = qlearning.train(maze, scheme, seed)
    finally:
        del maze.step                             # back to the normal Maze.step
    return r, _spy.Q, _spy.visits


def greedy_route(maze, Q):
    """Follow the greedy action from S without slip. Returns (cells, reached the goal)."""
    s, route, seen = maze.start, [maze.start], set()
    while s != maze.goal and not maze.is_trap[s] and s not in seen:
        seen.add(s)
        q = Q[s]
        s = maze.next_state[s][q.index(max(q))]
        route.append(s)
    return route, s == maze.goal


def route_name(maze, route):
    """Only meaningful for the Dyna mazes: does the route pass above the wall block (rows 0-1)?"""
    if not maze.name.startswith("dyna"):
        return "goal"
    cells = [divmod(s, maze.W) for s in route]
    return "upper" if any(r <= 1 and 2 <= c <= 6 for r, c in cells) else "lower"


# ---------------------------------------------------------------- choosing the schemes
def load_run(folder, algo, run):
    with open(os.path.join(folder, "evolution", f"{algo}_run{run}.json")) as f:
        return json.load(f)


def choose_schemes(maze, a):
    base = {b.name: b for b in baseline_schemes(maze)}
    panels = [("baseline", "Baseline A1: goal + small progress", base["A1_immediate_reward_only"]),
              ("baseline", "Baseline B1: + trap and wall penalties", base["B1_immediate_reward_punish"]),
              ("baseline", "Published: Russell & Norvig step cost", base["russell_norvig_matched"])]
    de = load_run(a.results, "de", a.run)
    me = load_run(a.results, "mapelites", a.run)
    panels.append(("DE", f"DE winner (run {a.run})", evolve.decode(de["best"]["genome"])))
    panels.append(("ME", f"MAP-Elites winner (run {a.run})", evolve.decode(me["best"]["genome"])))
    cell = list(a.elite_cell)
    elite = [e for e in me["archive"] if e["cell"] == cell]
    if elite:
        panels.append(("ME", f"MAP-Elites elite, map cell ({cell[0]}, {cell[1]})", evolve.decode(elite[0]["genome"])))
    else:
        print(f"map cell {cell} is empty in MAP-Elites run {a.run}; panel left out")
    return panels


def gene_line(s):
    parts = [f"trap {s.trap:g}" if s.trap == int(s.trap) else f"trap {s.trap:.1f}",
             f"wall {s.wall:.1f}", f"step {s.step:.2f}", f"progress {s.progress:.2f}"]
    if s.delay:
        parts.append(f"delay {s.delay}")
    return "   ".join(parts)


# ---------------------------------------------------------------- drawing
def draw_panel(ax, maze, title, scheme, group, visits, edges, n_seeds, summary, vmax):
    import matplotlib.patches as mpatches
    import numpy as np

    ax.set_xlim(-0.5, maze.W - 0.5)
    ax.set_ylim(maze.H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    levels = np.linspace(0, 1, len(SHADE) + 1)[1:]
    for r in range(maze.H):
        for c in range(maze.W):
            s = r * maze.W + c
            if (r, c) in maze.walls:
                fc = WALL
            elif (r, c) in maze.traps:
                fc = TRAP_FILL
            elif not vmax:
                fc = SHADE[1]
            else:
                v = (np.log1p(visits.get(s, 0)) / np.log1p(vmax)) if vmax else 0
                fc = SHADE[int(np.searchsorted(levels, min(v, 1.0)))] if v > 0 else SHADE[0]
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=fc,
                                            edgecolor=SURFACE, linewidth=1.5, zorder=1))
            if (r, c) in maze.traps:
                ax.plot([c - 0.22, c + 0.22], [r - 0.22, r + 0.22], color=TRAP_MARK, lw=1.6, zorder=2)
                ax.plot([c - 0.22, c + 0.22], [r + 0.22, r - 0.22], color=TRAP_MARK, lw=1.6, zorder=2)
    colour = GROUP_COLOUR[group]
    for (s1, s2), n in edges.items():
        (r1, c1), (r2, c2) = divmod(s1, maze.W), divmod(s2, maze.W)
        ax.plot([c1, c2], [r1, r2], color=colour, lw=0.6 + 5.4 * n / n_seeds, alpha=0.9,
                solid_capstyle="round", zorder=3)
    for s, label in ((maze.start, "S"), (maze.goal, "G")):
        r, c = divmod(s, maze.W)
        ax.text(c, r, label, ha="center", va="center", fontsize=10, fontweight="bold", color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE, edgecolor=colour, linewidth=1.4))
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=16)
    ax.text(0, 1.015, gene_line(scheme), transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
    ax.text(0, -0.03, summary, transform=ax.transAxes, fontsize=8, color=INK, va="top")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(ROOT, "results", "2026-11-15_laptop"))
    p.add_argument("--run", type=int, default=7, help="which run's winners to show")
    p.add_argument("--elite-cell", type=int, nargs=2, default=(5, 0), metavar=("SHARE", "DELAY"),
                   help="MAP-Elites map cell of the extra elite (default 5 0)")
    p.add_argument("--maze", default="dyna_9x6_3traps")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--shading", action="store_true", help="shade cells by visits in the first 100 episodes")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "figures"))
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maze = get_maze(a.maze)
    panels = choose_schemes(maze, a)
    seeds = range(SEEDS_START, SEEDS_START + a.seeds)
    rows, drawn = [], []
    for group, title, scheme in panels:
        visits, edges, routes, n90 = Counter(), Counter(), Counter(), []
        for seed in seeds:
            r, Q, v = train_and_read(maze, scheme, seed)
            visits.update(v)
            route, ok = greedy_route(maze, Q)
            routes[route_name(maze, route) if ok else "fails"] += 1
            if ok:
                edges.update(zip(route[:-1], route[1:]))
            n90.append(r["episodes_to_90"] if r["episodes_to_90"] is not None else evolve.PENALTY)
        visits = {s: n / a.seeds for s, n in visits.items()}
        med = statistics.median(n90)
        learned = sum(x < evolve.PENALTY for x in n90)
        if maze.name.startswith("dyna"):
            route_txt = f"upper {routes['upper']}, lower {routes['lower']}, fails {routes['fails']}"
        else:
            route_txt = f"reach the goal {routes['goal']}, fail {routes['fails']}"
        speed = f"learns in {med:.0f} episodes" if learned > a.seeds / 2 else f"{learned}/{a.seeds} seeds learn"
        summary = f"{speed}  ·  routes: {route_txt}"
        drawn.append((group, title, scheme, visits, edges, summary))
        rows.append(dict(panel=title, group=group, **{k: v for k, v in vars(scheme).items() if k != "source"},
                         median_episodes_to_learn=med, seeds_learned=learned,
                         route_upper=routes["upper"], route_lower=routes["lower"],
                         route_reaches_goal=routes["goal"], route_fails=routes["fails"],
                         cells_visited_early=len(visits)))
        print(f"{title:45s} {summary}")

    vmax = max(max(d[3].values(), default=0) for d in drawn) if a.shading else 0
    ncol = min(3, len(drawn))
    nrow = (len(drawn) + ncol - 1) // ncol
    # size the figure from the maze shape so there is no empty space (all sizes in inches)
    fig_w, left, right, wspace = 3.6 * ncol, 0.12, 0.12, 0.08
    ax_w = (fig_w - left - right) / (ncol + (ncol - 1) * wspace)
    ax_h = ax_w * maze.H / maze.W
    header, above, below, bottom = 0.62, 0.44, 0.30, 0.12
    fig_h = header + nrow * (above + ax_h + below) + bottom
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), facecolor=SURFACE, squeeze=False)
    fig.subplots_adjust(left=left / fig_w, right=1 - right / fig_w, wspace=wspace,
                        top=1 - (header + above) / fig_h, bottom=(bottom + below) / fig_h,
                        hspace=(above + below) / ax_h)
    for ax, (group, title, scheme, visits, edges, summary) in zip(axes.flat, drawn):
        draw_panel(ax, maze, title, scheme, group, visits, edges, a.seeds, summary, vmax)
    for ax in list(axes.flat)[len(drawn):]:
        ax.axis("off")
    note = "Line width = number of seeds whose final greedy policy takes that step.  x = trap.  "
    if a.shading:
        note += f"Shading = cells visited in the first {EARLY_EPISODES} training episodes (darker = more)."
    fig.text(left / fig_w, 1 - 0.10 / fig_h, f"Learned routes under different reward schemes "
             f"({maze.name}, {a.seeds} new seeds each)", fontsize=12, color=INK, va="top")
    fig.text(left / fig_w, 1 - 0.38 / fig_h, note, fontsize=8.5, color=MUTED, va="top")

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"maze_paths_{maze.name}")
    fig.savefig(stem + ".png", dpi=170, facecolor=SURFACE)
    fig.savefig(stem + ".pdf", facecolor=SURFACE)
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {stem}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
