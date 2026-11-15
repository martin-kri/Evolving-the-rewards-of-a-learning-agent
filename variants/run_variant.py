"""Run ONE evolutionary run (MAP-Elites or DE) with a different search range.

Same as run_evolution.py, plus --variant (see variant_defs.py for the list). The result
goes to results/variants/<variant>/evolution/ and also records the variant and its ranges.

Run from the repository folder:
    python variants/run_variant.py --variant wide_x2 --algo de --run 0 --budget 200 --quick   # smoke test
    python variants/run_variant.py --variant wide_x2 --algo mapelites --run 0
    python variants/run_variant.py --variant no_progress --algo de --run 0

--variant default gives exactly the same numbers as run_evolution.py.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from functools import partial

import variant_defs                      # also puts the repository folder on the import path
import evolve
from gridmaze import get_maze

TRAIN_MAZE = "dyna_9x6_3traps"


def _eval_one(x, maze_name, settings):
    return evolve.evaluate(x, get_maze(maze_name), **settings)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=list(variant_defs.VARIANTS))
    p.add_argument("--algo", choices=["mapelites", "de"], required=True)
    p.add_argument("--run", type=int, default=0, help="run seed (0, 1, 2, ...)")
    p.add_argument("--budget", type=int, default=5000, help="fitness evaluations (same for both algorithms)")
    p.add_argument("--maze", default=TRAIN_MAZE)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default=None, help="default: results/variants/<variant>/evolution")
    p.add_argument("--quick", action="store_true", help="fewer training episodes, for testing the code only")
    a = p.parse_args()

    variant_defs.apply(a.variant)                         # this (parent) process
    bounds = {g: list(v) for g, v in evolve.BOUNDS.items()}
    settings = dict(episodes=150) if a.quick else {}
    out = a.out or os.path.join(variant_defs.results_dir(a.variant), "evolution")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{a.algo}_run{a.run}.json")
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    log(f"start: variant {a.variant}, {a.algo}, run {a.run}, budget {a.budget} evaluations, maze {a.maze}, "
        f"{a.workers} workers{' (QUICK test mode)' if a.quick else ''}")
    log(f"gene ranges: {bounds}")

    # initializer: every worker process switches to the variant before it evaluates anything
    with mp.Pool(a.workers, initializer=variant_defs.apply, initargs=(a.variant,)) as pool:
        evaluate_batch = partial(pool.map, partial(_eval_one, maze_name=a.maze, settings=settings))
        if a.algo == "mapelites":
            archive, history = evolve.map_elites(evaluate_batch, a.budget, seed=a.run, log=log)
            result = dict(archive=[dict(cell=list(c), **ev) for c, ev in archive.items()])
            best = min(archive.values(), key=lambda e: e["fitness"])
        else:
            best, pop, history = evolve.differential_evolution(evaluate_batch, a.budget, seed=a.run, log=log)
            result = dict(population=pop)

    result.update(algo=a.algo, run=a.run, budget=a.budget, maze=a.maze, settings=settings,
                  variant=a.variant, bounds=bounds,
                  best=best, best_scheme=vars(evolve.decode(best["genome"])),
                  history=history, seconds=round(time.time() - t0))
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    log(f"saved {path}  best fitness {best['fitness']:.1f}  scheme {result['best_scheme']}")


if __name__ == "__main__":
    main()
