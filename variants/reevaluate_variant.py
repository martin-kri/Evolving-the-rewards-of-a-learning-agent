"""Re-evaluate the winners of one variant on fresh seeds (same procedure as reevaluate.py).

The genomes in the run files are stored in normalised units (0..1), so they only mean
something together with the gene ranges they were evolved with. This script reads the
ranges from the run files, switches evolve.py to them and then calls reevaluate.py.

Run from the repository folder:
    python variants/reevaluate_variant.py --variant wide_x2
    python variants/reevaluate_variant.py --variant wide_x2 --seeds 10 --mazes dyna_9x6_3traps   # quicker
    python variants/reevaluate_variant.py --dir results/2026-11-15_laptop      # any results folder works

Output: <folder>/reevaluation.csv, so the normal analysis works on it:
    python analyze.py --dir results/variants/wide_x2
"""
import argparse
import os
import sys

import variant_defs
import reevaluate


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--variant", choices=list(variant_defs.VARIANTS))
    g.add_argument("--dir", help="a results folder that contains evolution/*.json")
    a, extra = p.parse_known_args()        # everything else (--seeds, --mazes, --workers) goes to reevaluate.py

    folder = a.dir or variant_defs.results_dir(a.variant)
    runs = variant_defs.load_runs(folder)
    if not runs:
        raise SystemExit(f"no run files found in {os.path.join(folder, 'evolution')}")
    bounds = variant_defs.bounds_of_runs(runs)
    variant_defs.apply_bounds(bounds)
    print(f"{len(runs)} runs from {folder}")
    print(f"gene ranges: {bounds}")

    sys.argv = ["reevaluate.py", "--runs", os.path.join(folder, "evolution"),
                "--out", os.path.join(folder, "reevaluation.csv"), *extra]
    reevaluate.main()                      # decodes the winners in this process, with the ranges set above


if __name__ == "__main__":
    main()
