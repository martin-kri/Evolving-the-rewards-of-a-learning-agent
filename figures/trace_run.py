"""Re-run ONE evolutionary run and record EVERY evaluation, in order (the "trace").

The normal run files keep only the MAP-Elites archive or the final DE population. To draw
where DE searched, or how the MAP-Elites map filled up, we need every evaluated genome.
The runs are deterministic, so re-running with the same --run seed gives exactly the
same run as before; this script only adds the trace (it checks this against the saved
run file if one is given with --check).

Nothing in evolve.py is changed: the trace is collected by wrapping the evaluate_batch
function that the algorithms call.

Run from the repository folder:
    python figures/trace_run.py --algo de --run 7
    python figures/trace_run.py --algo mapelites --run 7
    python figures/trace_run.py --algo de --run 7 --check results/2026-11-15_laptop/evolution/de_run7.json

Output: results/figures/trace/<algo>_run<N>.json, same keys as a normal run file plus
"trace" (one entry per evaluation: genome, fitness, n90, converged, punishment_share,
mean_feedback_delay). About 1-2 MB per file.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evolve  # noqa: E402
from gridmaze import get_maze  # noqa: E402

TRAIN_MAZE = "dyna_9x6_3traps"


def _eval_one(x, maze_name, settings):
    return evolve.evaluate(x, get_maze(maze_name), **settings)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["mapelites", "de"], required=True)
    p.add_argument("--run", type=int, default=7, help="run seed; 7 is close to the median run for both algorithms")
    p.add_argument("--budget", type=int, default=5000)
    p.add_argument("--maze", default=TRAIN_MAZE)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default=os.path.join(ROOT, "results", "figures", "trace"))
    p.add_argument("--check", help="saved run file of the same run: confirm the re-run is identical")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.algo}_run{a.run}.json")
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    trace = []
    with mp.Pool(a.workers) as pool:
        batch = partial(pool.map, partial(_eval_one, maze_name=a.maze, settings={}))

        def evaluate_batch(genomes):            # same as run_evolution.py, plus recording
            evs = batch(genomes)
            trace.extend(evs)
            return evs

        if a.algo == "mapelites":
            archive, history = evolve.map_elites(evaluate_batch, a.budget, seed=a.run, log=log)
            result = dict(archive=[dict(cell=list(c), **ev) for c, ev in archive.items()])
            best = min(archive.values(), key=lambda e: e["fitness"])
        else:
            best, pop, history = evolve.differential_evolution(evaluate_batch, a.budget, seed=a.run, log=log)
            result = dict(population=pop)

    result.update(algo=a.algo, run=a.run, budget=a.budget, maze=a.maze, settings={},
                  best=best, best_scheme=vars(evolve.decode(best["genome"])),
                  history=history, seconds=round(time.time() - t0), trace=trace)

    if a.check:
        with open(a.check) as f:
            saved = json.load(f)
        same = saved["best"]["genome"] == best["genome"] and \
            [h[1] for h in saved["history"]] == [h[1] for h in history]
        log("check against the saved run: " + ("IDENTICAL" if same else "DIFFERENT (the trace is still valid "
                                                 "for this re-run, but it is not the saved run)"))
        result["identical_to"] = a.check if same else None

    with open(path, "w") as f:
        json.dump(result, f)
    log(f"saved {path}  ({len(trace)} evaluations, best fitness {best['fitness']:.1f})")


if __name__ == "__main__":
    main()
