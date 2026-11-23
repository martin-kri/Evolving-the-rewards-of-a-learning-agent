# Scaling: how far do reward schemes from the small maze scale?

Step 3 of the plan in the main README. A reward scheme is a fixed set of numbers. To test it on
a bigger maze, a **new agent** (empty Q-table) learns that maze with those rewards, and we count
the episodes until it succeeds 90% of the time. Nothing is evolved here.

The bigger mazes are copies of the training maze `dyna_9x6_3traps` scaled 2x to 5x (every cell
becomes a k x k block, see `gridmaze.scale_layout`), so only the size changes, not the layout:

| Size | Maze | Shortest safe path | Training episodes |
| --- | --- | --- | --- |
| 1x | `dyna_9x6_3traps` | 16 | 500 |
| 2x | `dyna_18x12_3traps` | 31 | 1000 |
| 3x | `dyna_27x18_3traps` | 45 | 1500 |
| 4x | `dyna_36x24_3traps` | 60 | 2000 |
| 5x | `dyna_45x30_3traps` | 74 | 2500 (not in the default run) |

For every scheme this gives its **size limit**: the biggest copy it still learns.

## How to run

From the repository folder:

```
python scaling/transfer.py --runs 7 --seeds 3                                # pilot, ~5 min
python scaling/transfer.py                                                   # all saved single-maze runs
python scaling/transfer.py --dirs results/2026-11-15_laptop results/multimaze/A --name single_vs_multi
python scaling/transfer.py --resume                                          # continue an interrupted job
python scaling/transfer.py --summarize results/scaling/transfer/transfer.csv  # tables only
```

Output in `results/scaling/<name>/`:
- `transfer.csv`: one row per scheme, size and seed (genes, episodes to 90%, learned or not,
  final success, trap falls). Written while the runs finish, so `--resume` can continue.
- `summary.csv`: one row per scheme: genes, share of seeds that learned and median episodes per
  size, and the size limit. This is the input for step 4 (the trap-vs-goal rule) and the figures.

The printed table groups the schemes (winners and elites per condition and algorithm, and every
hand-made scheme) and shows, per size, the share of schemes that still learn and their median
episodes.

## Choices (for now), alternatives, and how to change them

1. **Which schemes.** The winner of every run, the elites of the MAP-Elites and random-search maps
   that learned the training maze on every evaluation seed, and the six main hand-made schemes.
   Why: a scheme that does not learn the small maze cannot scale, and skipping those saves time.
   Alternatives: every elite (`--elites all`), winners only (`--elites none`), every hand-made
   scheme (`--baselines all`), a subset of runs (`--runs 0 1 2`).
2. **Which runs.** Any results folder: the saved single-maze runs (default), multi-maze runs
   (`results/multimaze/A`), variants with other gene ranges (their own ranges are used to decode
   the genomes). Folders are compared side by side with `--dirs`.
3. **Sizes 1x to 4x.** Why: in the pilot, half of the elites already fail at 3x, and 4x (path 60)
   still costs little. Alternatives: add 5x (`--scales 1 2 3 4 5`); sizes in between (1.5x,
   2.5x) with `gridmaze.scale_layout_proportional`, for a finer curve; scaled copies of other
   layouts (e.g. `maze_8x8`), to see whether the limits depend on the layout. The last two need
   new entries in `gridmaze.SCALED` / `LAYOUTS` and in `SCALED` here.
4. **Training episodes = 500 x size.** Why: a k times longer path needs more exploration, and with
   a proportional budget every size gets the same exploration per step of path (epsilon decays
   over the episodes). Alternatives: the same number for every size (`--fixed-episodes 1500`, as
   in the re-evaluation on the `scaled-mazes` branch); a budget that grows with the number of
   cells (k x k), which is more generous for big mazes (small code change).
5. **"Learns a size" = at least 80% of the seeds reach 90% success** (8 of 10 seeds). Why: one or
   two unlucky seeds should not decide it. Alternatives: every seed (`--threshold 1.0`), most
   seeds (`--threshold 0.5`).
6. **Size limit = the biggest size learned, with every smaller size learned too.** Why: a scheme
   that fails at 2x but passes 3x by luck should not get limit 3. Alternative: the biggest size
   learned at all (small code change in `summarize`).
7. **10 new seeds per scheme and size** (6000 and up; never used before), the same seeds for every
   scheme, so schemes are compared on the same runs. Why: cost; the re-evaluation uses 30. For
   the final report: `--seeds 20` or 30 for the winners and baselines (`--elites none`).
8. **Hand-made schemes are rebuilt for each size.** The "matched" ones (e.g.
   `russell_norvig_matched`) have their step cost matched to the path length of the maze they
   run on. Why: that is how they are defined. Alternative: keep their 1x numbers on every size,
   like the evolved schemes (small code change), which tests them under the same conditions.

## Pilot (run 7 only, 3 seeds, 1x to 4x)

| Group | Schemes | Learn 1x | 2x | 3x | 4x | Median size limit |
| --- | --- | --- | --- | --- | --- | --- |
| single-maze ME winner | 1 | yes (120) | yes (210) | yes (250) | yes (250) | 4 |
| single-maze DE winner | 1 | yes (140) | yes (150) | yes (220) | yes (300) | 4 |
| single-maze ME elites | 53 | 94% | 81% | 49% | 30% | 2 |
| hand-made B1 | 1 | yes (280) | yes (140) | yes (280) | yes (320) | 4 |
| hand-made A1 | 1 | yes (260) | yes (290) | yes (390) | yes (360) | 4 |
| hand-made Russell & Norvig matched | 1 | yes (270) | yes (670) | yes (1180) | yes (1720) | 4 |
| sparse and delayed-only (A2, B2) | 3 | no | no | no | no | 0 |

Numbers in brackets: median episodes to 90%. Only 3 seeds, so "yes" means 3 of 3. 732 training
runs took 5.5 minutes on 2 cores; the full default run (all 10 single-maze runs, 10 seeds) is
about 30x that.
