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

# Step 4: the trap-vs-goal rule (`rule.py`)

Step 3 measures **whether** a scheme still learns a bigger maze. Step 4 asks **why not**, from the
rewards alone: does the scheme still make the goal the best way to end an episode? If a walk into
a trap (or wandering until the time limit) earns more than the long way to the goal, no agent
can learn to reach the goal, however long it trains. If the goal is still best and the agent
fails anyway, the reason is learning speed (exploration, too few episodes), not the incentive.

The idea comes from the lecture (Russell & Norvig 4x3 world, the optimal policy as a function of
the step reward): with a step cost, the reward of the long path to the goal shrinks with its
length, and at some point ending the episode early is better. Our version also has discounting,
progress rewards, delays and walls, so the simple comparison of the lecture is not enough and
`rule.py` computes it exactly.

## Three ways to compute it

| | Method | What it includes | Status |
| --- | --- | --- | --- |
| 1 | **Simple rule** (the lecture's comparison): return of the shortest safe path to the goal vs return of the shortest path into the nearest trap | step cost and discount; not the progress reward's detours, walls, the time limit, or slip | not implemented (the intuition only) |
| 2 | **Exact best path in the maze without slip** (deterministic moves) | every reward exactly as in `qlearning.train()`: goal, trap, wall, step, progress, delays (also "end"), the MiniGrid goal, gamma = 0.95, the time limit (10 x shortest path); all paths, also detours and wandering | **implemented, `rule.py`** |
| 3 | **Same with the 10% slip** (the agent's real world) | as 2, plus random moves: risky paths next to traps cost something | not implemented (later) |

Method 2 is what the code computes, so it is the method the report describes. Method 1 is only
the motivation (the lecture slide); method 3 is the natural next step if method 2 leaves too many
failures unexplained.

How method 2 works: for every scheme and size it finds the best return of each possible ending
(goal, trap, time limit) over **all** paths from the start, by dynamic programming over the steps
left until the end. Delays are exact: a reward earned j steps before the end of the episode is paid
at step tau - 1 - max(j - k, 0) (rewards still pending at the end are paid at the last step, as in
`train()`). Checked against a brute-force search over every path on a small maze (40 random
schemes with every kind of delay: identical up to 4e-15).

Prediction: the scheme "can learn" a size if the goal is its best ending there. The **margin** is
the best goal return minus the best other return (how much the goal is worth more). The
**predicted size limit** is the biggest size where the goal is best, with every smaller size too
(the same definition as the observed size limit in step 3).

How to add the other two:
- Method 1: BFS distance from the start to the nearest trap (`gridmaze` has the BFS), then the
  closed-form returns of the two straight paths with the same delay formula. A few lines; useful
  to show in the report how often the simple rule and the exact one disagree.
- Method 3: the same dynamic programming with expected values over the slip (prob. 0.9 + 0.025
  intended move, 0.025 each other direction). Exact for schemes without delay. With delays the
  end time is random, so the formula above no longer holds; options: ignore the delay for
  method 3, or estimate the best policy's return by simulation. The prediction would then be "the
  best policy reaches the goal in at least 90% of the episodes", which is exactly our success
  criterion.

## How to run

```
python scaling/rule.py                                                    # saved single-maze runs, 1x-5x, ~4 min
python scaling/rule.py --transfer results/scaling/transfer/transfer.csv    # + compare with step 3
python scaling/transfer.py --runs 7 --seeds 3 --name pilot_run7                                # the pilot, ~5 min
python scaling/rule.py --runs 7 --scales 1 2 3 4 --baselines main --transfer results/scaling/pilot_run7/transfer.csv --name rule_pilot7
```

Use the same `--dirs`, `--runs` and `--elites` as the `transfer.py` run it is compared with.
Output in `results/scaling/<name>/`: `rule.csv` (one row per scheme and size: best return of each
ending, best path lengths, best ending, margin) and `rule_summary.csv` (one row per scheme:
genes, best ending and margin per size, predicted and observed size limit).

## First results (saved single-maze runs; comparison only with the pilot)

All 20 winners, 533 learning elites and 12 hand-made schemes, 1x to 5x:

| Group | Schemes | Goal best at 1x | 2x | 3x | 4x | 5x |
| --- | --- | --- | --- | --- | --- | --- |
| single-maze winners (ME and DE) | 20 | 100% | 100% | 100% | 100% | 100% |
| single-maze ME elites | 533 | 100% | 99% | 95% | 88% | 81% |
| hand-made | 12 | 100% | 100% | 100% | 100% | 92% |

- The goal's margin shrinks by the discount, about 5% per extra step of path: for most hand-made
  schemes it is about 4.7 at 1x and 0.24 at 5x. So on big mazes the goal is still best, but only
  barely.
- The one hand-made scheme that flips is `A2_delayed_reward_only` at 5x: everything is paid at the
  end, so the whole sum is discounted by the full path length. A walk of 34 steps towards the
  goal that ends in a trap on the way (0 for the trap, +0.1 per step of progress, return 0.63)
  is worth more than the 74 steps to the goal (return 0.41). At 4x the goal still wins, by 0.07.
- Compared with the pilot transfer run (run 7, 3 seeds, 1x to 4x): whenever the goal was **not**
  the best ending, the agents never learned (0 of 2 at 3x, 0 of 8 at 4x), so the rule did not
  give a wrong "fails". But most failures happen although the goal is best (4x: 53 schemes with
  the goal best, only 21 learned). Those failures are about learning speed: compared with the
  schemes that learned, they have bigger step costs, almost no progress reward, longer delays and
  much milder traps; their margin is about the same (4x: 0.45 vs 0.50). The milder traps would
  matter only with slip (method 3).
- The sparse and delayed-only hand-made schemes have the goal as the best ending at every size,
  but do not learn even 1x in the pilot: an example of the right incentive with too little
  guidance.

To do with the full transfer run: the same comparison over all runs and 10 seeds, then decide
whether method 3 is needed.
