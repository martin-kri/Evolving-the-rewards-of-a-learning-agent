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

# Step 6: drift as a test condition

The agent drifts: with probability `slip` its move is replaced by another one. Until now that was
always 0.1, in our style. Step 6 tests the saved reward schemes in a world with **more drift**, and
in the lecture's style, as a second test condition next to size. Together with step 3 this gives a
"size x drift" picture for every scheme: which sizes it still learns at which drift.

## Drift styles (`gridmaze.py`)

| Style | With probability `slip`, the move becomes ... | `slip` 0.1 | `slip` 0.2 |
| --- | --- | --- | --- |
| `uniform` (ours, default) | a random one of the 4 directions | 92.5% intended, 2.5% each side, 2.5% back | 85% intended, 5% each other direction |
| `sideways` (the lecture, Russell & Norvig 4x3 world) | one of the 2 directions at a right angle | 90% intended, 5% each side | **80% intended, 10% each side** (the lecture's numbers) |

`get_maze(name, slip=0.1, drift="uniform")`. The default is unchanged, and so is every earlier
result (checked: identical training runs, and the transfer pilot is reproduced exactly).

`qlearning.train(..., train_slip=P)` trains at drift P while the tests (success rate, N90) still
use the maze's own drift. Default `None`: both the same, as before.

## Can the world still be learned at all?

More drift can make a maze impossible: even a perfect policy then falls into a trap too often.
`Maze.best_success(time_limit)` computes the best possible chance of reaching the goal in time
(exact dynamic programming over the drift), so we can tell "the scheme failed" from "nobody can
pass". For the scaled training maze, with the normal time limit (10 x shortest path):

| Drift | 1x | 2x | 3x | 4x | 5x |
| --- | --- | --- | --- | --- | --- |
| uniform 0.1 (normal) | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 |
| uniform 0.2 | 0.981 | 0.999 | 1.000 | 1.000 | 1.000 |
| uniform 0.3 | 0.949 | 0.996 | 1.000 | 1.000 | 1.000 |
| uniform 0.4 | **0.892** | 0.988 | 0.999 | 1.000 | 1.000 |
| sideways 0.1 to 0.3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

- Up to uniform 0.3, every size can be passed (above 90%), so a failure is the scheme's (or the
  agent's), not the world's. Uniform 0.4 makes the small maze impossible.
- Bigger copies are **safer**: their corridors are k cells wide, so the agent can keep a distance
  from the traps. So with drift the small maze can be harder than the big ones, the opposite of
  the size effect without drift.
- With sideways drift a perfect policy loses nothing: it can choose a route where a sideways slip
  never leads into a trap (the time limit leaves room for detours). The learned policy still has
  to find that route.

`transfer.py` prints this "best possible success" under every table.

## How to run

```
python scaling/transfer.py --elites none --drift uniform:0.1 uniform:0.2 uniform:0.3 sideways:0.2 --name drift_winners
python scaling/transfer.py --drift uniform:0.1 uniform:0.2 uniform:0.3 sideways:0.2 --name drift              # + elites (server)
python scaling/transfer.py --elites none --train-slip 0.3 --name trained_at_0.3                                 # learn at 0.3, tested at 0.1
python scaling/transfer.py --summarize results/scaling/drift/transfer.csv --limit biggest                       # other size limit
```

The same seeds (6000 and up) are used for every drift, so the conditions are compared on the same
runs. Every drift condition gets its own table, then a "size x drift" table: per group, the median
size limit at every drift and (in brackets) the share of schemes that learn each size. The CSV
has the columns `drift`, `slip` and `train_slip`; files from before step 6 are read as uniform 0.1.

Cost: each drift condition costs as much as one transfer run. Winners and hand-made schemes only
(`--elites none`, 26 schemes) with 4 drifts are about 4,200 training runs, roughly half an hour
on 2 cores. With the learning elites (about 550 schemes) it is about 20 times that: server.

## Choices (for now), alternatives, and how to change them

1. **Drift levels: uniform 0.1 (reference), 0.2, 0.3, and sideways 0.2 (the lecture).** Why: up to
   0.3 every size can still be passed (table above); sideways 0.2 is the lecture's world.
   Alternatives: finer steps (0.15, 0.25), sideways 0.1 / 0.3, uniform 0.4 (impossible at 1x),
   drift 0 (no slip). Change: the list after `--drift`.
2. **The agent learns and is tested at the same drift.** Why: the question is whether the reward
   scheme still teaches the agent in a harder world, like the size question. Alternative:
   `--train-slip 0.1` with a higher `--drift`: learned in the calm world, judged in the drifting
   one (robustness of the learned policy). The reverse (`--train-slip 0.3`, tested at 0.1) is the
   "training drift" idea from the plan (slower but safer).
3. **"Learns" = at least 80% of the seeds reach 90% success, as without drift.** Why: the same
   rule everywhere, and up to uniform 0.3 a perfect policy gets at least 95%. Alternative: a
   threshold relative to the best possible success (e.g. 90% of it), fairer at high drift; small
   code change (pass `success_threshold` to `train()` per maze).
4. **Size limit: the biggest size learned with every smaller size learned too (as in step 3).**
   With drift this can mislead: a scheme that fails 1x but learns 2x (the small maze is the risky
   one) gets limit 0. So the size x drift table also shows the share that learns each size, and
   `--limit biggest` uses the biggest size learned at all.
5. **Same drift style everywhere else.** Evolution, re-evaluation and screening stay at uniform
   0.1. Evolving the training drift (a gene, with safety as a second objective) is in "Later".

## First check (not a result)

Run 7's two winners and the six main hand-made schemes, 3 seeds, 1x and 2x (median episodes to
90%; "no" = fewer than 80% of the seeds learned):

| Scheme | uniform 0.1: 1x / 2x | sideways 0.2: 1x / 2x | uniform 0.3: 1x / 2x |
| --- | --- | --- | --- |
| ME winner | 120 / 210 | 360 / 240 | no / 520 |
| DE winner | 140 / 150 | no / 390 | no / no |
| B1 | 280 / 140 | 280 / 230 | no / 390 |
| A1 | 260 / 290 | no / 920 | no / no |
| Russell & Norvig matched | 270 / 670 | 270 / 600 | no / 700 |
| A2, B2, sparse | no / no | no / no | no / no |

Already visible with 3 seeds: at uniform 0.3 nobody learns the small maze (best possible 0.95),
but some schemes learn the 2x copy; the two winners react differently to drift (ME better than
DE); the trap-punishing B1 and Russell & Norvig keep up best. With `--train-slip 0.3` (tested at
0.1) everything that learned before still learns, more slowly (DE winner 140 -> 220 at 1x).
