"""Run ONE search run (MAP-Elites, DE or random search) with the fitness measured on a SET of
mazes instead of one (see training_sets.py).

Same algorithms, genome, budget and output format as run_evolution.py; only the fitness
changes. The result goes to results/multimaze/<set>/evolution/<algo>_run<N>.json and also
records the training set, so the normal re-evaluation and analysis work on it.

Run from the repository folder:
    python multimaze/run_multimaze.py --algo mapelites --run 0 --budget 200 --quick     # smoke test
    python multimaze/run_multimaze.py --algo mapelites --run 0                          # full run, set A
    python multimaze/run_multimaze.py --algo de --run 0
    python multimaze/run_multimaze.py --algo random --run 0
    python multimaze/run_multimaze.py --set B --algo mapelites --run 0                  # later, server

--set single gives exactly the same numbers as run_evolution.py (a check of this code).

Options for screening (see screening/):
    --combine mean|relative|worst    how the mazes are combined into one fitness (training_sets.py)
    --variant NAME                   other gene ranges, from variants/variant_defs.py
Results of non-default options go to their own folder, e.g. results/multimaze/A_worst_wide_progress/.
Runs with a --variant must be re-evaluated with variants/reevaluate_variant.py (it reads the
gene ranges from the run files).
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from functools import partial

import sys

import training_sets                     # also puts the repository folder on the import path
import evolve

sys.path.insert(0, os.path.join(training_sets.ROOT, "variants"))
import variant_defs  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="A", choices=list(training_sets.SETS), help="training set (default A)")
    p.add_argument("--algo", choices=["mapelites", "de", "random"], required=True)
    p.add_argument("--run", type=int, default=0, help="run seed (0, 1, 2, ...)")
    p.add_argument("--budget", type=int, default=5000, help="fitness evaluations (same for all algorithms)")
    p.add_argument("--combine", default="mean", choices=training_sets.COMBINE,
                   help="how the mazes are combined into one fitness (default mean)")
    p.add_argument("--variant", default="default", choices=list(variant_defs.VARIANTS),
                   help="gene ranges (default: the normal ones)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default=None, help="default: results/multimaze/<set>[_<combine>][_<variant>]/evolution")
    p.add_argument("--quick", action="store_true", help="150 training episodes per run, for testing the code only")
    a = p.parse_args()

    entries = training_sets.entries_of(a.set)
    variant_defs.apply(a.variant)                         # this (parent) process
    bounds = {g: list(v) for g, v in evolve.BOUNDS.items()}
    tag = a.set + ("" if a.combine == "mean" else f"_{a.combine}") + ("" if a.variant == "default" else f"_{a.variant}")
    condition = f"multi-maze {a.set}" + ("" if a.combine == "mean" else f", {a.combine}") \
        + ("" if a.variant == "default" else f", {a.variant}")
    if a.set == "single":
        condition = condition.replace("multi-maze single", "single-maze")
    out = a.out or os.path.join(training_sets.ROOT, "results", "multimaze", tag, "evolution")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{a.algo}_run{a.run}.json")
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    runs = sum(len(s) for _, _, s in entries)
    log(f"start: training set {a.set} ({runs} training runs per evaluation: "
        f"{', '.join(m for m, _, _ in entries)}), {a.algo}, run {a.run}, budget {a.budget}, "
        f"combine {a.combine}, gene ranges {a.variant}, {a.workers} workers{' (QUICK test mode)' if a.quick else ''}")
    log("first progress line after the initial population (can take a few minutes) ...")

    # initializer: every worker process switches to the gene ranges of the variant (on Windows the
    # workers re-import evolve.py with the default ranges)
    with mp.Pool(a.workers, initializer=variant_defs.apply, initargs=(a.variant,)) as pool:
        evaluate_batch = partial(pool.map, partial(training_sets.evaluate_on_set, set_name=a.set,
                                                   quick=a.quick, combine=a.combine))
        if a.algo == "mapelites":
            archive, history = evolve.map_elites(evaluate_batch, a.budget, seed=a.run, log=log)
            best = min(archive.values(), key=lambda e: e["fitness"])
        elif a.algo == "random":
            best, archive, history = evolve.random_search(evaluate_batch, a.budget, seed=a.run, log=log)
        else:
            best, pop, history = evolve.differential_evolution(evaluate_batch, a.budget, seed=a.run, log=log)
        if a.algo == "de":
            result = dict(population=pop)
        else:
            result = dict(archive=[dict(cell=list(c), **ev) for c, ev in archive.items()])

    result.update(algo=a.algo, run=a.run, budget=a.budget, maze=f"training set {a.set}",
                  settings=dict(quick=True) if a.quick else {},
                  condition=condition, training_set=a.set, combine=a.combine,
                  variant=a.variant, bounds=bounds,
                  training_mazes=[dict(maze=m, episodes=e, seeds=list(s)) for m, e, s in entries],
                  best=best, best_scheme=vars(evolve.decode(best["genome"])),
                  history=history, seconds=round(time.time() - t0))
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    per_maze = ", ".join(f"{m} {sum(v) / len(v):.0f}" for m, v in best["per_maze"].items())
    log(f"saved {path}  best fitness {best['fitness']:.1f}  (per maze: {per_maze})")


if __name__ == "__main__":
    main()
