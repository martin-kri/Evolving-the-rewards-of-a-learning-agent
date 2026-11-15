# Variants: trying other search ranges

Side experiments for two open design questions. Nothing in the main experiment is changed:
these scripts import the files of the repository and only overwrite the gene ranges
(`evolve.BOUNDS`) at run time. If we decide to keep a variant, the only change needed in
the main code is the `BOUNDS` table in `evolve.py`.

**Question 1: are the gene ranges wide enough?** In the first full experiment 6 of the 10 DE
winners had the progress gene at the upper edge of its range (0.95 or more out of 1.0).
A best value at the edge of the searched interval usually means the interval should be
extended until the best value is inside it.

**Question 2: how much of the result depends on the progress reward?** The progress reward
pays the agent for getting closer to the goal, and "closer" is measured with the true
shortest-path distance, so it gives away part of the solution. The `no_progress` variant
switches it off and asks what evolution finds with goal, trap, wall, step and delay only.

## The variants

| name | trap | wall | step | progress | why |
|---|---|---|---|---|---|
| `default` | -20..0 | -5..0 | -1..0 | 0..1 | the main experiment (control) |
| `wide_progress` | -20..0 | -5..0 | -1..0 | **0..3** | widen only the gene that sat at the edge |
| `wide_x2` | **-40..0** | **-10..0** | **-2..0** | **0..2** | everything 2x wider |
| `wide_x4` | **-80..0** | **-20..0** | **-4..0** | **0..4** | everything 4x wider |
| `no_progress` | -20..0 | -5..0 | -1..0 | **off** | question 2 |

The goal reward stays +10 and the delay gene stays 0..30 steps in every variant (the delay
range is also the size of the delay axis of the MAP-Elites map). To try another range, add
one line to `VARIANTS` in `variant_defs.py`.

## What widening a range does

- **More room.** Schemes with stronger rewards become possible. If the best scheme lies
  outside the old range, only a wider range can find it.
- **Thinner search.** The budget (5000 evaluations) stays the same but the space is larger,
  so every region is sampled less densely.
- **Coarser steps.** Genes are stored in 0..1 and mutated in those units (MAP-Elites
  sigma = 0.1 of the range; DE differences likewise). A 4x wider range means every
  mutation is 4x larger in real reward units, so fine-tuning is harder.
- **More failures.** Most of the added space is bad. Of 200 random schemes, 36% learn the
  training maze on all seeds with the default range, 25% with `wide_x2`, 20% with `wide_x4`.
  The main reason is the step cost: below about -1 per step, walking to the goal costs
  more than the goal pays.
- **Not comparable with old runs.** A wider range is a different experiment, so both
  algorithms and the re-evaluation have to be run again. The baselines do not change.

## Step 0: the cheap test (about 5 minutes, no evolution)

```
python variants/sweep_genes.py
```

Takes the 20 winners of `results/2026-11-15_laptop`, sets one gene at a time to fixed
values up to 4x outside the searched range, and retrains on new seeds. Output in
`results/variants/sweep/` (`sweep.png`, `sweep_summary.csv`).

Result of 2026-11-15 (median episodes to learn on the training maze; as evolved: ME 125, DE 120):

| gene | inside the searched range | outside |
|---|---|---|
| progress | 0 is bad (235 to 260), best around 0.75 to 1.0 (102 to 118) | DE winners a little faster at 1.5 to 2 (102), no gain after that; ME winners no gain |
| step | flat from 0 to -0.75; ME winners already worse at -1 | learning breaks: at -2 only 51 to 73% of the runs learn, at -3 about 20% |
| trap | 0 is bad (many runs never learn); -5 to -20 all similar | flat down to -80 |
| wall | flat | slower at -10 and -20 |

Reading: only the progress gene shows a (small) gain outside its range. A wider step range
adds only failing schemes; wider trap and wall ranges add nothing. So `wide_progress` is
the variant this test supports; `wide_x2` and `wide_x4` are the "check that we are not
missing something" variants, because this test changes one gene at a time and cannot see
combinations.

### A first small screening (2026-11-15, not saved in the repository)

DE only, 3 runs per variant, budget 1000 (a fifth of the real budget), winners retrained on
30 fresh seeds. Median episodes to learn:

| variant | training maze | maze 8x8 | maze 10x10 | winners: step / progress (median) |
|---|---|---|---|---|
| `default` | 130 | 100 | 180 | -0.60 / 0.68 |
| `wide_progress` | 105 | 80 | 130 | -0.95 / 2.62 |
| `wide_x2` | 110 | 95 | 150 | -1.38 / 1.67 |
| `wide_x4` | 120 | 100 | 180 | -2.33 / 3.74 |

With 3 runs none of the differences is significant (p = 0.35 for the two best variants
against `default`), so this is a hint and not a result. Two things to note: the wider
variants are not worse even at a small budget, and their winners combine a larger progress
reward with a larger step cost. The sweep above tests one gene at a time and could not
find that combination.

## Steps 1 to 4: a full experiment for one variant

Run everything from the repository folder. Example for `wide_progress`:

```
# 1. evolution (same options as run_evolution.py)
#    PowerShell
0..9 | ForEach-Object { python variants/run_variant.py --variant wide_progress --algo mapelites --run $_ }
0..9 | ForEach-Object { python variants/run_variant.py --variant wide_progress --algo de --run $_ }
#    bash
for r in 0 1 2 3 4 5 6 7 8 9; do python variants/run_variant.py --variant wide_progress --algo mapelites --run $r; done
for r in 0 1 2 3 4 5 6 7 8 9; do python variants/run_variant.py --variant wide_progress --algo de --run $r; done

# 2. winners and baselines on fresh seeds
python variants/reevaluate_variant.py --variant wide_progress

# 3. the normal analysis (statistics, QD metrics, plots)
python analyze.py --dir results/variants/wide_progress

# 4. all variants next to the default experiment
python variants/compare_variants.py
```

Results go to `results/variants/<variant>/` (ignored by git, like all of `results/`,
until we decide to keep a snapshot).

Smoke test of the code (about a minute):

```
python variants/run_variant.py --variant wide_x2 --algo de --run 0 --budget 200 --quick
```

## How much to run

A full experiment (10 + 10 runs, budget 5000) takes 4 to 5 hours per variant on the laptop.
Two cheaper options:

- **Screening**: 3 runs per algorithm with `--budget 2000` (about 40 minutes per variant).
  Enough to drop a variant that is clearly worse, not enough to claim a small improvement.
  Run `default` with the same reduced settings as the control, because a comparison is only
  fair at the same budget:
  `python variants/run_variant.py --variant default --algo de --run 0 --budget 2000`
  (this writes to `results/variants/default/`, not into the main results).
- **Server**: the full protocol for the one or two variants that survive the screening.

## Deciding

- Keep the **default** range if no wider variant is clearly faster on fresh seeds. Then the
  sweep figure and one sentence in the report are the evidence that the range was checked.
- Adopt a **wider** range if its winners are faster on fresh seeds on the training maze
  (table 2 of `compare_variants.py`: p value and "chance faster") and do not get worse on
  the unseen mazes (table 1), and the winners are no longer at the edge (table 3).
- **no_progress** is not a candidate to replace the main experiment. It is an extra result:
  how good can a reward scheme get without the distance hint, and does the map look
  different.

## Notes

- `--variant default` reproduces `run_evolution.py` exactly (checked: same best genome and
  history for the same run seed).
- Genomes in the run files are stored in 0..1, so they only mean something together with
  their ranges. `run_variant.py` saves the ranges in every run file (`variant`, `bounds`)
  and `reevaluate_variant.py` reads them from there. Do not use the plain `reevaluate.py`
  on a variant folder: it would decode the winners with the default ranges.
- For the same reason `plot_archive.py --gene ...` shows wrong gene values for a variant.
  The fitness map without `--gene` is fine.
- Keep one variant per folder.
