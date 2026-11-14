"""Re-evaluate the evolved winners and the hand-made baselines on FRESH seeds.

Why: during evolution every genome is scored on only 5 seeds, and picking the best of
thousands of noisy scores also picks lucky ones ("winner's curse"). The fair comparison
retrains every candidate from scratch on seeds never used during evolution.

What is compared, on every maze in --mazes:
    DE best             the best genome of each DE run
    ME best             the best elite of each MAP-Elites run
    baselines           the hand-made and published schemes from rewards.baseline_schemes()

The training maze answers question 1 (sample efficiency); the other mazes, never seen
during evolution, answer question 2 (generalisation).

Example:
    python reevaluate.py --runs results/evolution --seeds 30
Output: results/reevaluation.csv, one row per (maze, candidate, seed).
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os

import evolve
from gridmaze import get_maze
from qlearning import train
from rewards import baseline_schemes

TEST_MAZES = ["dyna_9x6_3traps",                       # training maze
              "dyna_9x6_2traps", "dyna_9x6",            # same layout, different traps
              "maze_8x8", "maze_10x10", "lava_7x5"]     # unseen layouts
FRESH_SEEDS_START = 1000                                # never used during evolution


def _job(args):
    maze_name, label, scheme, seed = args
    r = train(get_maze(maze_name), scheme, seed)
    return dict(maze=maze_name, candidate=label, scheme=scheme.name, seed=seed,
                episodes_to_90=r["episodes_to_90"] if r["episodes_to_90"] is not None else evolve.PENALTY,
                converged=r["episodes_to_90"] is not None, final_success=r["final_success"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="results/evolution")
    p.add_argument("--mazes", nargs="+", default=TEST_MAZES)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--out", default="results/reevaluation.csv")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    candidates = []                                      # (label, RewardScheme)
    for path in sorted(glob.glob(os.path.join(a.runs, "*.json"))):
        with open(path) as f:
            run = json.load(f)
        label = "ME best" if run["algo"] == "mapelites" else "DE best"
        candidates.append((label, evolve.decode(run["best"]["genome"], name=f"{run['algo']}_run{run['run']}")))

    seeds = range(FRESH_SEEDS_START, FRESH_SEEDS_START + a.seeds)
    jobs = []
    for m in a.mazes:
        maze = get_maze(m)
        cands = candidates + [("baseline", s) for s in baseline_schemes(maze)]
        jobs += [(m, label, s, seed) for label, s in cands for seed in seeds]
    print(f"{len(jobs)} training runs on {a.workers} workers ...")
    with mp.Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=4)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
