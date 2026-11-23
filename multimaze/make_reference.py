"""Reference difficulty of every training maze, for --combine relative.

Reference = median episodes to 90% (on the 500-episode scale) that the 20 single-maze winners
(10 MAP-Elites + 10 DE, results/2026-11-15_laptop) need on that maze, over 5 new seeds
(7000-7004). With --combine relative, a scheme's N90 on each maze is divided by that maze's
reference, so a hard maze counts as much as an easy one.

Run from the repository folder (about 2 minutes); writes multimaze/reference_n90.json:
    python multimaze/make_reference.py
"""
import glob
import json
import multiprocessing as mp
import os
import statistics

import training_sets                      # also puts the repository folder on the import path
import evolve
from qlearning import train

SEEDS = range(7000, 7005)
WINNERS = os.path.join(training_sets.ROOT, "results", "2026-11-15_laptop", "evolution")


def _job(args):
    maze_name, episodes, genome = args
    scheme = evolve.decode(genome)
    out = []
    for s in SEEDS:
        n = train(training_sets._maze(maze_name), scheme, s, episodes=episodes)["episodes_to_90"]
        out.append((n if n is not None else 2 * episodes) * training_sets.BASE_EPISODES / episodes)
    return maze_name, statistics.median(out)


def main():
    genomes = [json.load(open(p))["best"]["genome"] for p in sorted(glob.glob(os.path.join(WINNERS, "*_run*.json")))]
    mazes = {}
    for entries in training_sets.SETS.values():
        for m, e, _ in entries:
            mazes[m] = e
    jobs = [(m, e, g) for m, e in mazes.items() for g in genomes]
    print(f"{len(genomes)} winners x {len(mazes)} mazes x {len(SEEDS)} seeds ...")
    per = {m: [] for m in mazes}
    with mp.Pool() as pool:
        for m, v in pool.imap_unordered(_job, jobs):
            per[m].append(v)
    ref = {m: statistics.median(v) for m, v in per.items()}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_n90.json")
    with open(path, "w") as f:
        json.dump(dict(description="median N90 (500-episode scale) of the 20 single-maze winners, "
                                   "seeds 7000-7004; made by make_reference.py", reference=ref), f, indent=1)
    for m, v in ref.items():
        print(f"  {m:20s} {v:6.0f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
