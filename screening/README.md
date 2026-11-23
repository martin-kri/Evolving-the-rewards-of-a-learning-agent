# Screening: try many configurations, run only the best ones in full

A full experiment is 10 runs x 3 algorithms x 5000 evaluations per configuration (about 7 hours
on the laptop for the single maze, 2.4x that for `A2`). We have many configurations to choose
from (training set, how the mazes are combined, gene ranges, ...), so we first try all of them
**cheaply**, look at the results, and make full runs only for the one or two that look best.

## Validation mazes vs test mazes

If we chose the configuration by its results on the test mazes and then reported its results on
the same mazes, we would make the winner's curse one level up: some configuration is always lucky
on the mazes it was chosen with. So the choice is made on five extra random mazes that are used
for nothing else:

| Maze | Size | Shortest safe path | Traps |
| --- | --- | --- | --- |
| `valid_7x6` | 7x6 | 15 | 2 |
| `valid_9x8` | 9x8 | 17 | 4 |
| `valid_10x7` | 10x7 | 17 | 4 |
| `valid_11x10` | 11x10 | 21 | 5 |
| `valid_13x9` | 13x9 | 26 | 6 |

Made like the other random mazes (`gridmaze.random_maze`, 22% walls, seeds noted in
`gridmaze.LAYOUTS`), a bit bigger on average than the training mazes of set A (6x6 to 12x8), since
we care about transfer to bigger mazes. They are never used for training. The `unseen_*` mazes
and the scaled copies are **not** used in screening, so the final comparison on them stays clean.

| Mazes | Used for |
| --- | --- |
| `dyna_9x6_3traps`, `train_*` | training (fitness during evolution) |
| `valid_*` | choosing between configurations (this folder) |
| `unseen_*`, scaled copies, hand-made mazes | final results, only for the chosen configurations |

## Protocol (chosen for now)

1. **Run**: every configuration, MAP-Elites and DE, 3 runs each (run seeds 0, 1, 2), budget
   2000 evaluations.
2. **Re-evaluate**: the winner of every run and the hand-made schemes are retrained on each
   validation maze with 10 fresh seeds (1000-1009), 500 episodes, as in `reevaluate.py` (through
   `variants/reevaluate_variant.py`, which decodes the winners with the gene ranges of their runs).
3. **Compare**: one number per run winner, its **validation score** = mean episodes to 90% over
   the validation mazes and seeds (1000 for a run that never gets there). Per configuration and
   algorithm: the median over the 3 runs, min-max, the share of re-evaluation runs that learned,
   the per-maze medians, map coverage (MAP-Elites), and a Mann-Whitney test with A12 against the
   reference configuration (`single`, the original experiment).
4. **Decide** (by us, looking at the table): drop configurations that are clearly worse than
   `single` (A12 near 0, or winners that fail on some validation maze); among the rest, take the
   one or two with the best median, and the cheaper one when two are close.
5. **Full runs** (server): 10 runs x 3 algorithms x 5000 for the chosen ones, then
   `reevaluate.py` and `scaling/transfer.py` on the test mazes as for the main experiment.

With 3 runs, only big differences can show up as significant (with 3 vs 3 the smallest possible
p-value is 0.1). Screening is for dropping bad options and finding candidates, not a result for
the report.

## Configurations (`configs.py`)

| Name | Training set | Combine | Gene ranges | Question | Default |
| --- | --- | --- | --- | --- | --- |
| `single` | single (original maze) | mean | default | reference | yes |
| `A` | A (original + 5 random mazes, 1 run each) | mean | default | does a set of mazes help? | yes |
| `A2` | A with 2 runs per maze | mean | default | does less noise in the fitness help? | yes |
| `A_relative` | A | relative | default | does it help if every maze counts the same? | yes |
| `A_worst` | A | worst | default | does it help to optimise the slowest maze? | yes |
| `A_wide` | A | mean | `wide_progress` (progress 0-3) | is the progress range too narrow? | yes |
| `single_wide` | single | mean | `wide_progress` | same, without the set | |
| `single_noprog`, `A_noprog` | single / A | mean | `no_progress` | how much does the progress gene matter? | |
| `B` | A + the 2x copy (1000 episodes) | mean | default | does a bigger maze in the set help? (server) | |

Combine (see `multimaze/README.md`): `mean` = plain average of all runs (what the papers do);
`relative` = each maze's mean divided by what the single-maze winners need there
(`multimaze/reference_n90.json`), so the big mazes do not dominate; `worst` = the slowest maze.
A note on `worst`: most random schemes fail on at least one maze, so at the start almost every
scheme gets 1000 and the search has little to go on. It may still work once a few schemes learn
everything; screening will show it.

To add a configuration, add a line to `CONFIGS` (and a variant to `variants/variant_defs.py` if it
needs new gene ranges).

## Time

`python screening/run_screening.py list` prints the estimate. With the default selection (6
configurations, 2 algorithms, 3 runs, budget 2000) it is about **4 hours** on the laptop (from the
saved runs: 5000 evaluations take about 17 minutes for MAP-Elites and 5 for DE), plus a few
minutes of re-evaluation per configuration. With random search too, about 7 hours. On a faster
machine, `list --speed 0.25` scales the estimate.

## How to run

From the repository folder (PowerShell):

```
python screening/run_screening.py list                         # configurations and time
python screening/run_screening.py all                          # run + re-evaluate + compare
python screening/run_screening.py all --configs single A A2    # only some (the first is the reference)
python screening/run_screening.py compare --score relative     # another way to score, no new runs
python screening/run_screening.py all --runs 2 --budget 120 --seeds 2 --quick --out results/screening_test   # code test, ~4 min
```

Every step skips what already exists (a run file, a `reevaluation.csv`), so it can be stopped and
started again, and more runs or configurations can be added later. After adding runs to a
configuration, redo its re-evaluation with `reeval --force`.

On the server, `run --print-commands` prints one command per run, to start them in parallel (one
SLURM array task each), then `reeval` and `compare` at the end.

Output in `results/screening/` (ignored by git):

- `<configuration>/evolution/<algo>_run<N>.json`: the runs (same format as `multimaze/`)
- `<configuration>/reevaluation.csv`: winners and hand-made schemes on the validation mazes
- `comparison_<score>.csv`: the table that `compare` prints

## Choices, alternatives, and how to switch

| Choice | Now | Alternatives | How to switch |
| --- | --- | --- | --- |
| Runs per configuration | 3 | 5 (more reliable, 5/3 of the time) | `--runs 5` (adds runs 3 and 4 to the existing ones) |
| Budget | 2000 | 5000 (as the full runs), 1000 (faster, but MAP-Elites has barely left its first random genomes) | `--budget` (use a new `--out`: run files do not say they are a screening run) |
| Algorithms | MAP-Elites, DE | + random search | `--algos mapelites de random` |
| Validation mazes | 5 random, 7x6 to 13x9 | more mazes, bigger ones, the 2x copy | `VALIDATION_MAZES` in `configs.py` (new mazes in `gridmaze.LAYOUTS`) |
| Validation seeds | 10 per maze | 30 as `reevaluate.py` (3x the re-evaluation time) | `--seeds 30` with `reeval --force` |
| Score | mean over the validation mazes (the biggest maze weighs most) | `relative`: each maze relative to the best hand-made scheme there, then averaged (1 = as fast, every maze weighs the same) | `compare --score relative` (no new runs needed) |
| Reference | `single` | any configuration | put it first in `--configs` |

A caveat about the budget: configurations are ranked after 2000 evaluations, but the full runs
have 5000. A configuration that improves slowly but keeps improving (e.g. `A2`, where each
evaluation is less noisy) could look worse at 2000 than it would at 5000. If two configurations
are close, check their `history` (best fitness over evaluations) in the run files.
