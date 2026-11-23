# Multi-maze evolution

Step 2 of the plan in the main README. In the main experiment every reward scheme is scored on
one maze (`dyna_9x6_3traps`), and the evolved schemes are specialists for that maze: they beat
the hand-made schemes there, but mostly not on unseen mazes, and on the 3x bigger copy the
hand-made B1 is as fast. Here the fitness is measured on a **set of mazes of different sizes and
layouts**, so evolution has to find schemes that work across them, as in Singh et al. (2010) and
Niekum (2010), who measure fitness over a distribution of environments.

Question: do schemes evolved on several small mazes learn bigger and unseen mazes faster than
the single-maze schemes and the hand-made ones?

Nothing else changes: same genome and ranges, same algorithms (MAP-Elites, DE, random search),
same budget of 5000 evaluations, same output format. The existing runs in
`results/2026-11-15_laptop` are the **single-maze** condition.

## The training set (set A)

The original training maze plus five random mazes of different sizes, one training run each:

| Maze | Size | Shortest safe path | Traps | Seed of the run |
| --- | --- | --- | --- | --- |
| `dyna_9x6_3traps` | 9x6 | 16 | 3 | 0 |
| `train_6x6` | 6x6 | 12 | 2 | 1 |
| `train_8x6` | 8x6 | 14 | 3 | 2 |
| `train_8x8` | 8x8 | 16 | 4 | 3 |
| `train_10x8` | 10x8 | 18 | 5 | 4 |
| `train_12x8` | 12x8 | 20 | 6 | 5 |

- The random mazes come from `gridmaze.random_maze` (22% walls) and are stored as text in
  `gridmaze.LAYOUTS`, so they never change. For every size, the first seed was kept where a trap
  lies on a shortest route (the path through traps is shorter than the safe path), so the traps
  matter as in the Dyna maze. The seeds are noted next to each layout.
- Different sizes on purpose: the trap-vs-goal rule says schemes with step costs break when the
  path gets long, so evolution should see a range of path lengths.
- The original maze stays in the set, so both conditions have trained on it and the comparison
  on its scaled copies (step 3) is fair.
- Selection pressure grows with size: of 10 random reward schemes, 6 learn `train_6x6` within 500
  episodes, 2 learn `train_8x8`, 1 learns `train_10x8` and none learns `train_12x8`. The evolved
  winners of the single-maze runs learn all of them (about 100 to 270 episodes).

Five more random mazes are **never used for training** and are test mazes in `reevaluate.py`
(6x8 to 12x10, shortest paths 14 to 23): `unseen_6x8`, `unseen_7x7`, `unseen_9x7`, `unseen_11x8`,
`unseen_12x10`.

## Fitness over a set

- Every (maze, seed) pair is one training run with the normal settings (500 episodes).
- Fitness = mean over all runs of the episodes needed to reach 90% (N90), with 1000 for a run that
  never gets there. That is the same scale as the single-maze fitness, so the numbers can be
  compared and `analyze.py` (QD-score = 1000 - fitness) works unchanged.
- A maze trained for more episodes (only in set B) has its N90 scaled to the 500-episode scale
  (x 500 / episodes), so every run counts the same.
- This plain average is what the papers do: Singh et al. (2010) use the expected fitness over the
  distribution of environments, Niekum (2010) the fitness across a distribution of environments,
  Zheng et al. (2020) and Oh et al. (2020) average over lifetimes and tasks, and GERS the average
  return over its validation environments. Alternatives (each maze relative to a reference such as
  B1, or the worst maze) are possible later as a `--combine` option.
- Behaviour descriptors (punishment share, feedback delay) are averaged over all runs: the
  MAP-Elites map has the same axes as before.
- The run file also stores the N90 per maze of every evaluated scheme (`per_maze`).

Cost: 6 training runs per evaluation instead of 5, measured at 1.21x the time of the single-maze
fitness (a MAP-Elites run: about 21 instead of 17 minutes on the laptop).

## Other sets

- `single`: the original experiment (one maze, seeds 0-4). Gives exactly the same results as
  `run_evolution.py` (checked, also with the Windows way of starting worker processes).
- `A2`: set A with 2 training runs per maze (12 per evaluation, seeds 0-11). Less noise in the
  fitness (the random part shrinks by about 30%), which should reduce the winner's curse; about
  2x the cost of A (a MAP-Elites run: about 42 minutes on the laptop). For screening: A vs A2.
- `B`: set A plus the 2x copy of the training maze (`dyna_18x12_3traps`, 1000 episodes, 2 seeds).
  About 2.5x the cost. For later, on the server.

To make a new set, add an entry to `SETS` in `training_sets.py`.

## How to run

From the repository folder (PowerShell):

```
python multimaze/run_multimaze.py --algo mapelites --run 0 --budget 200 --quick     # smoke test, ~20 s

0..9 | ForEach-Object { python multimaze/run_multimaze.py --algo mapelites --run $_ }
0..9 | ForEach-Object { python multimaze/run_multimaze.py --algo de --run $_ }
0..9 | ForEach-Object { python multimaze/run_multimaze.py --algo random --run $_ }

python reevaluate.py --runs results/multimaze/A/evolution --out results/multimaze/A/reevaluation.csv
python analyze.py --dir results/multimaze/A
```

Results go to `results/multimaze/<set>/` (ignored by git, like all of `results/`).

To compare single-maze and multi-maze winners, both have to be re-evaluated on the same mazes and
seeds. The saved snapshot was re-evaluated before the scaled and the unseen random mazes existed,
so re-evaluate a copy of its runs (the snapshot itself stays unchanged):

```
New-Item -ItemType Directory -Force results/single/evolution
Copy-Item results/2026-11-15_laptop/evolution/*_run*.json results/single/evolution/
python reevaluate.py --runs results/single/evolution --out results/single/reevaluation.csv
python variants/compare_variants.py --dirs results/single results/multimaze/A
```

`compare_variants.py` prints the fresh-seed learning speed of the winners on every test maze, a
test against the first folder, the map coverage, and the winners' genes.

## First check (not a result)

One MAP-Elites and one DE run on set A with a budget of only 300 evaluations, re-evaluated on 5
fresh seeds, next to the 10 single-maze runs with the full budget of 5000 (median episodes to
learn):

| | training maze | `unseen_7x7` | `unseen_11x8` |
| --- | --- | --- | --- |
| single-maze ME (10 runs, 5000) | 105 | 125 | 345 |
| single-maze DE (10 runs, 5000) | 100 | 100 | 295 |
| multi-maze ME (1 run, 300) | 150 | 130 | 280 |
| multi-maze DE (1 run, 300) | 110 | 100 | 320 |

Only a check that everything runs: one run each, 6% of the budget, 5 seeds. The real comparison
needs the full runs (server).
