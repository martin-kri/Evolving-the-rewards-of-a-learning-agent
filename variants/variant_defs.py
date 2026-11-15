"""Search-space variants: the same experiment with different gene ranges.

A variant only changes evolve.BOUNDS (the real-value range of each gene). The genome,
the fitness, the algorithms, the budget and the MAP-Elites axes stay the same, so a
variant can be compared with the default experiment.

The files of the main experiment are NOT modified. A variant is applied at run time
by overwriting evolve.BOUNDS in the current process (see apply()).

To try another range, add one line to VARIANTS and use its name with --variant.
"""
import copy
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository folder
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evolve  # noqa: E402

DEFAULT_BOUNDS = copy.deepcopy(evolve.BOUNDS)
REWARD_GENES = ("trap", "wall", "step", "progress")   # "delay" is never widened, see below


def widened(factor, genes=REWARD_GENES):
    """Default range of each listed gene multiplied by `factor` (the 0 end stays at 0)."""
    return {g: tuple(v * factor for v in DEFAULT_BOUNDS[g]) for g in genes}


# name -> the genes whose range differs from the default
VARIANTS = {
    # the experiment as it is in evolve.py: trap -20..0, wall -5..0, step -1..0, progress 0..1
    "default": {},

    # every reward gene 2x / 4x wider: trap -40 / -80, wall -10 / -20, step -2 / -4, progress 2 / 4
    "wide_x2": widened(2),
    "wide_x4": widened(4),

    # only the progress reward wider (the gene where the DE winners sat at the edge)
    "wide_progress": {"progress": (0.0, 3.0)},

    # no progress reward at all: the gene is switched off (range 0..0), so evolution has to
    # find a scheme made of goal, trap, wall, step and delay only
    "no_progress": {"progress": (0.0, 0.0)},
}
# The delay gene keeps its range in every variant: MAX_DELAY is also the size of the delay
# axis of the MAP-Elites map, so changing it would change the map and break the comparison.


def bounds_of(name):
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant '{name}'. Choose one of: {', '.join(VARIANTS)}")
    b = copy.deepcopy(DEFAULT_BOUNDS)
    b.update(VARIANTS[name])
    return b


def apply_bounds(bounds):
    """Overwrite evolve.BOUNDS in THIS process. evolve.decode() reads it at call time."""
    evolve.BOUNDS.clear()
    evolve.BOUNDS.update({g: tuple(v) for g, v in bounds.items()})


def apply(name):
    """Switch this process to a named variant. Also used as the start-up function of the
    worker processes: on Windows every worker re-imports evolve.py with the default range,
    so the parent's change would otherwise be lost."""
    apply_bounds(bounds_of(name))


def results_dir(name):
    return os.path.join(ROOT, "results", "variants", name)


def load_runs(folder):
    """All run files (*.json) of one results folder, e.g. results/variants/wide_x2."""
    evo = os.path.join(folder, "evolution")
    if not os.path.isdir(evo):
        evo = folder
    runs = []
    for path in sorted(glob.glob(os.path.join(evo, "*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def bounds_of_runs(runs):
    """The gene ranges the runs were evolved with (default if the file does not say).
    Stops if the folder mixes runs with different ranges."""
    found = {}
    for r in runs:
        b = {g: tuple(v) for g, v in r.get("bounds", DEFAULT_BOUNDS).items()}
        found[json.dumps(b, sort_keys=True)] = b
    if len(found) > 1:
        raise SystemExit("the run files in this folder were evolved with different gene ranges; "
                         "keep one variant per folder")
    return next(iter(found.values())) if found else copy.deepcopy(DEFAULT_BOUNDS)


if __name__ == "__main__":
    for name in VARIANTS:
        print(f"{name:14s}", {g: tuple(v) for g, v in bounds_of(name).items()})
