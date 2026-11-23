"""Training sets for multi-maze evolution, and the fitness of a reward scheme over a set.

Evolving on one maze produces specialists for that maze. Here the fitness of a scheme is
measured on a SET of mazes of different sizes and layouts, so evolution has to find schemes
that work across them (as in Singh et al. 2010 and Niekum 2010, where fitness is measured
over a distribution of environments).

A set is a list of entries (maze name in gridmaze.LAYOUTS, training episodes, seeds). Every
(maze, seed) pair is one training run. The fitness is the mean over all pairs of the episodes
needed to reach 90% (N90), with 2 x episodes for a run that never gets there. A maze that is
trained for more than the default 500 episodes has its N90 scaled to the 500-episode scale
(x 500 / episodes), so every pair counts the same and the "never learned" value is always 1000,
as in the single-maze experiment.

The behaviour descriptors (punishment share, feedback delay) are averaged over all pairs, so
the MAP-Elites map has the same axes as before.

Sets:
    single  the original experiment: dyna_9x6_3traps, seeds 0-4. Gives exactly the same
            results as run_evolution.py (used to check this code).
    A       dyna_9x6_3traps plus five random mazes of different sizes (6x6 to 12x8, shortest
            paths 12 to 20), one training run each: 6 runs per evaluation, about 1.2x the
            cost of the single-maze fitness.
    B       A plus the 2x copy of the training maze (1000 episodes, 2 seeds): about 2.5x the
            cost. For later, on the server.

The random mazes are stored as text in gridmaze.LAYOUTS (train_*), together with three random
mazes that are never used for training (unseen_*), for testing.
"""
import functools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the repository folder
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evolve  # noqa: E402
from gridmaze import get_maze  # noqa: E402
from qlearning import DEFAULTS, train  # noqa: E402

BASE_EPISODES = DEFAULTS["episodes"]          # 500: the scale every N90 is reported on

_A = [("dyna_9x6_3traps", 500, (0,)),
      ("train_6x6", 500, (1,)),
      ("train_8x6", 500, (2,)),
      ("train_8x8", 500, (3,)),
      ("train_10x8", 500, (4,)),
      ("train_12x8", 500, (5,))]

SETS = {
    "single": [("dyna_9x6_3traps", 500, (0, 1, 2, 3, 4))],
    "A": _A,
    "B": _A + [("dyna_18x12_3traps", 1000, (6, 7))],
}


@functools.lru_cache(maxsize=None)
def _maze(name):
    return get_maze(name)


def entries_of(set_name):
    if set_name not in SETS:
        raise SystemExit(f"unknown training set '{set_name}'. Choose one of: {', '.join(SETS)}")
    return SETS[set_name]


def evaluate_on_set(x, set_name, quick=False):
    """Fitness and descriptors of genome x over a training set. Same keys as
    evolve.evaluate(), plus per_maze (the scaled N90 of every run, by maze).
    quick=True trains every run for 150 episodes only (for testing the code)."""
    scheme = evolve.decode(x)
    n90, learned, share, delay, per_maze = [], 0, [], [], {}
    for maze_name, episodes, seeds in entries_of(set_name):
        eps = 150 if quick else episodes
        scale = BASE_EPISODES / eps
        values = []
        for seed in seeds:
            r = train(_maze(maze_name), scheme, seed, episodes=eps)
            ok = r["episodes_to_90"] is not None
            values.append((r["episodes_to_90"] if ok else 2 * eps) * scale)
            learned += ok
            share.append(r["punishment_share"])
            delay.append(r["mean_feedback_delay"])
        n90 += values
        per_maze[maze_name] = values
    return dict(
        genome=list(x),
        fitness=sum(n90) / len(n90),
        n90=n90,
        converged=learned,
        punishment_share=sum(share) / len(share),
        mean_feedback_delay=sum(delay) / len(delay),
        per_maze=per_maze,
    )


if __name__ == "__main__":
    for name, entries in SETS.items():
        runs = sum(len(s) for _, _, s in entries)
        print(f"{name:7s} {runs} training runs per evaluation: " +
              ", ".join(f"{m} ({e} episodes, seeds {list(s)})" for m, e, s in entries))
