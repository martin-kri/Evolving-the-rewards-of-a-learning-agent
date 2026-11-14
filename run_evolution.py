"""Run ONE evolutionary run (MAP-Elites or DE) and save it to one JSON file.

One run = one algorithm + one run seed, so many runs can go in parallel
(several terminals, or a SLURM job array on the HPC).

Examples:
    python run_evolution.py --algo mapelites --run 0 --budget 200 --quick   # smoke test, ~1 min
    python run_evolution.py --algo mapelites --run 0                        # full run
    python run_evolution.py --algo de --run 0
    for r in 0 1 2 3 4 5 6 7 8 9; do python run_evolution.py --algo de --run $r; done

SLURM (one job per run seed):
    #SBATCH --array=0-9
    #SBATCH --cpus-per-task=8
    python run_evolution.py --algo mapelites --run $SLURM_ARRAY_TASK_ID --workers 8
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from functools import partial

import evolve
from gridmaze import get_maze

TRAIN_MAZE = "dyna_9x6_3traps"


def _eval_one(x, maze_name, settings):
    return evolve.evaluate(x, get_maze(maze_name), **settings)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["mapelites", "de"], required=True)
    p.add_argument("--run", type=int, default=0, help="run seed (0, 1, 2, ...)")
    p.add_argument("--budget", type=int, default=5000, help="fitness evaluations (same for both algorithms)")
    p.add_argument("--maze", default=TRAIN_MAZE)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default="results/evolution")
    p.add_argument("--quick", action="store_true", help="fewer training episodes, for testing the code only")
    a = p.parse_args()

    settings = dict(episodes=150) if a.quick else {}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.algo}_run{a.run}.json")
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    log(f"start: {a.algo}, run {a.run}, budget {a.budget} evaluations, maze {a.maze}, "
        f"{a.workers} workers{' (QUICK test mode)' if a.quick else ''}")
    log("first progress line after the initial population (can take a minute or two) ...")

    with mp.Pool(a.workers) as pool:
        evaluate_batch = partial(pool.map, partial(_eval_one, maze_name=a.maze, settings=settings))
        if a.algo == "mapelites":
            archive, history = evolve.map_elites(evaluate_batch, a.budget, seed=a.run, log=log)
            result = dict(archive=[dict(cell=list(c), **ev) for c, ev in archive.items()])
            best = min(archive.values(), key=lambda e: e["fitness"])
        else:
            best, pop, history = evolve.differential_evolution(evaluate_batch, a.budget, seed=a.run, log=log)
            result = dict(population=pop)

    result.update(algo=a.algo, run=a.run, budget=a.budget, maze=a.maze, settings=settings,
                  best=best, best_scheme=vars(evolve.decode(best["genome"])),
                  history=history, seconds=round(time.time() - t0))
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    log(f"saved {path}  best fitness {best['fitness']:.1f}  scheme {result['best_scheme']}")


if __name__ == "__main__":
    main()
