# Evolving the rewards of a learning agent

Bio-Inspired Artificial Intelligence project: Can Quality-Diversity evolution (MAP-Elites) discover a diverse archive of intrinsic reward functions that significantly improve reinforcement learning sample efficiency compared to hand-crafted rewards?

## Idea

A tabular Q-learning agent learns to cross a small maze with walls and traps. Instead of
hand-designing its rewards, we evolve them. MAP-Elites (Lecture 12) searches over reward schemes
(how much the goal, traps, wall bumps, steps and progress are worth, and how late the feedback
arrives). Every cell of the map keeps the scheme that teaches fastest for one "teaching style":
how much of the feedback is punishment, and how delayed it is.

The biological inspiration: evolution decided what animals find rewarding or painful, and
Skinner's operant conditioning describes how they learn from those signals during a lifetime.
Here the outer loop (evolution) designs the rewards, the inner loop (Q-learning) learns from them.

We compare the evolved map with our own hand-made 2x2 baseline (reward vs reward + punishment,
immediate vs delayed) and with reward schemes from textbooks and benchmark papers.


## Status

- [x] Maze environment, tabular Q-learning agent, baseline reward schemes, experiment runner
- [x] Final training maze: Sutton & Barto's 9x6 Dyna Maze with 3 traps (`dyna_9x6_3traps`)
- [x] Stable N90 metric (moving average over the last 3 tests)
- [x] MAP-Elites and Differential Evolution over reward schemes (`evolve.py`, `run_evolution.py`)
- [x] Re-evaluation on fresh seeds and unseen mazes, statistics and plots (`reevaluate.py`, `analyze.py`)
- [ ] Full experiment: 10 MAP-Elites + 10 DE runs (HPC)
- [ ] Report

## Quick start

Python 3.10+. The agent and the evolution use only the standard library; numpy, scipy and
matplotlib are needed for the statistics and the plots.

```
pip install -r requirements.txt
python run_baselines.py --quick
```

Full experiment, in this order:

```
python run_evolution.py --algo mapelites --run 0 --budget 200 --quick   # smoke test, ~30 s
0..9 | ForEach-Object { python run_evolution.py --algo mapelites --run $_ }   # PowerShell
0..9 | ForEach-Object { python run_evolution.py --algo de --run $_ }
0..9 | ForEach-Object { python run_evolution.py --algo random --run $_ }   # non-evolutionary baseline
python plot_archive.py results/evolution/mapelites_run0.json
python reevaluate.py --runs results/evolution --seeds 30
python analyze.py
```

On a SLURM cluster, `sbatch run_all.slurm` launches all 20 runs as one job array.

## Files

| File | What it does |
| --- | --- |
| `gridmaze.py` | The maze, built from text (walls, traps, slipping, shortest paths); named layouts in `LAYOUTS` and `random_maze()` |
| `rewards.py` | `RewardScheme` (goal, trap, wall, step, progress, delay) and `baseline_schemes()` |
| `qlearning.py` | Tabular Q-learning (`train`) with fixed settings in `DEFAULTS`, and the metrics |
| `run_baselines.py` | Runs mazes x schemes x seeds in parallel, saves CSVs and plots to `results/` |
| `evolve.py` | Reward genome (5 genes, goal fixed at +10), fitness, behaviour descriptors, MAP-Elites, Differential Evolution and random search |
| `run_evolution.py` | One search run (`--algo mapelites`, `de` or `random`, plus a run seed) from the command line, saved to `results/evolution/<algo>_run<N>.json` |
| `run_all.slurm` | SLURM job array: all 10 MAP-Elites + 10 DE runs in parallel on a cluster |
| `plot_archive.py` | Heatmap of a MAP-Elites archive (fitness or a gene value per cell) |
| `reevaluate.py` | Retrains the evolved winners and the baselines on 30 fresh seeds, on the training maze and on unseen mazes |
| `variants/`, `figures/`, `multimaze/`, `scaling/`, `screening/` | Side experiments and figures, each with its own README |
| `analyze.py` | Summary tables, Mann-Whitney U tests with Holm correction (one test per evolved scheme), Vargha-Delaney A12, quality-diversity metrics (cells filled, QD-score), boxplots, convergence curves |

## Mazes

Written as text rows: `.` free, `#` wall, `T` trap, `S` start, `G` goal. Stepping into a trap ends
the episode; bumping into a wall leaves the agent in place; with 10% probability the agent slips
in a random direction.

| Name | Size | Shortest safe path | Origin |
| --- | --- | --- | --- |
| `lava_7x5` | 7x5 | 8 | DeepMind AI Safety Gridworlds, lava world |
| `dyna_9x6` | 9x6 | 14 | Sutton & Barto, Dyna Maze (Example 8.1) |
| `dyna_9x6_2traps` | 9x6 | 16 | Dyna Maze + 2 traps (ours) |
| `dyna_9x6_3traps` | 9x6 | 16 | Dyna Maze + 3 traps (ours) |
| `maze_8x8` | 8x8 | 14 | ours |
| `maze_10x10` | 10x10 | 18 | ours |
| `dyna_18x12_3traps` | 18x12 | 31 | `dyna_9x6_3traps` scaled x2 (every cell a 2x2 block) |
| `dyna_27x18_3traps` | 27x18 | 45 | `dyna_9x6_3traps` scaled x3 (every cell a 3x3 block) |
| `train_6x6` ... `train_12x8` | 6x6 to 12x8 | 12 to 20 | random mazes, training set of `multimaze/` |
| `unseen_6x8` ... `unseen_12x10` (5 mazes) | 6x8 to 12x10 | 14 to 23 | random mazes, never used for training (test) |
| `valid_7x6` ... `valid_13x9` (5 mazes) | 7x6 to 13x9 | 15 to 26 | random mazes, never used for training; only for choosing between configurations (`screening/`) |

The scaled copies keep the layout of the training maze and change only its size, so they separate
"the maze is bigger" from "the maze is different". `scale_layout(rows, k)` makes them (walls and traps
become k x k blocks; S and G stay single cells at the same relative position), and
`scale_layout_proportional(rows, H, W)` scales to any size by mapping coordinates proportionally.
Bigger mazes need more episodes: re-evaluate them with `--episodes 1500`.

## Reward schemes

All schemes pay +10 for the goal. For tabular Q-learning starting from all-zero Q-values, only the
ratios between rewards matter.

| Field | Meaning |
| --- | --- |
| `goal` | reaching the goal (ends the episode) |
| `trap` | stepping into a trap (ends the episode) |
| `wall` | bumping into a wall or the edge |
| `step` | every step (negative = step cost) |
| `progress` | +w per step closer to the goal, -w per step away, 0 when standing still |
| `delay` | `0` = immediate; `k` = each reward arrives k steps later; `"end"` = all rewards summed and paid at the last step |
| `minigrid_goal` | goal pays 10 x (1 - 0.9 x t / time limit), as in MiniGrid |

Baselines in `baseline_schemes()`:

| Scheme | Goal | Trap | Wall | Step | Progress | Delay | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `A1_immediate_reward_only` | +10 | 0 | 0 | 0 | 0.1 | 0 | ours |
| `A2_delayed_reward_only` | +10 | 0 | 0 | 0 | 0.1 | end | ours |
| `B1_immediate_reward_punish` | +10 | -5 | -1 | 0 | 0.1 | 0 | ours (Taxi and Sutton & Barto ratios) |
| `B2_delayed_reward_punish` | +10 | -5 | -1 | 0 | 0.1 | end | ours |
| `B1_trap10` | +10 | -10 | -1 | 0 | 0.1 | 0 | ours, trap = -goal |
| `sparse_dyna` | +10 | 0 | 0 | 0 | 0 | 0 | Sutton & Barto Dyna Maze, FrozenLake |
| `kenton` | +10 | -10 | 0 | 0 | 0 | 0 | Kenton et al. 2019 |
| `russell_norvig` | +10 | -10 | 0 | -0.4 | 0 | 0 | Russell & Norvig 4x3 world |
| `russell_norvig_matched` | +10 | -10 | 0 | size-matched | 0 | 0 | Russell & Norvig, step cost scaled to maze size |
| `ai_safety` | +10 | -10 | 0 | -0.2 | 0 | 0 | AI Safety Gridworlds lava world |
| `ai_safety_matched` | +10 | -10 | 0 | size-matched | 0 | 0 | AI Safety lava world, step cost scaled to maze size |
| `minigrid` | time-dependent | 0 | 0 | 0 | 0 | 0 | MiniGrid |

"Size-matched" scales the published step cost so the shortest path costs the same share of the goal
as in the original maze (Russell & Norvig: 5 steps; AI Safety lava world: 8 steps).

## Agent and metrics

Fixed for every scheme (`DEFAULTS` in `qlearning.py`): 500 episodes, learning rate 0.2, discount 0.95,
epsilon-greedy from 1.0 to 0.05 over the first 70% of episodes, time limit 10 x shortest path.
Every 10 episodes the greedy policy is tested 20 times.

| Metric | Meaning |
| --- | --- |
| `episodes_to_90`, `steps_to_90` | training episodes / environment steps until the **moving average of the last 3 tests** (30 episodes) reaches 90% success (`stable_window` in `DEFAULTS`; 1 = first single test, which is noisy) |
| `final_success` | test success after training |
| `trap_falls`, `wall_bumps` | counted during training |
| `punishment_share` | share of all feedback received that was negative (planned MAP-Elites axis) |
| `mean_feedback_delay` | average steps between an event and its reward (planned MAP-Elites axis) |

Outputs in `results/`: `summary.csv`, `runs.csv`, `overview.png` (median episodes to 90% for every
maze and scheme), `curves_<maze>.png`, `mazes.png`.
Why a moving average: during training the greedy policy oscillates, so a single lucky test could
reach 90% long before the agent has really learned (e.g. B1 on `dyna_9x6_2traps`, seed 1: 95% at
episode 80, then 45% for the next 140 episodes). Since this metric is the fitness of the
evolution, the noisy version would reward luck.


## Evolving reward schemes

**Genome.** Five genes, each searched in a normalised [0, 1] range and decoded by `evolve.decode()`:

| Gene | Range |
| --- | --- |
| `trap` | -20 ... 0 |
| `wall` | -5 ... 0 |
| `step` | -1 ... 0 |
| `progress` | 0 ... 1 |
| `delay` | 0 ... 30 steps (integer; 0 = immediate) |

The goal reward is fixed at +10: for tabular Q-learning from all-zero Q-values only the ratios
between rewards matter, so a free goal gene would be redundant.

**Fitness** (minimised): mean `episodes_to_90` over 5 training seeds (0-4, the same for every
genome), with a penalty of 1000 (= 2 x 500 episodes) for each seed that never reaches 90%.

**MAP-Elites** (Mouret & Clune 2015): 10x10 archive over punishment share x mean feedback delay,
both measured during training. 100 random genomes, then batches of 50 Gaussian mutations
(sigma = 0.1) of random elites.

**Differential Evolution** (Storn & Price 1997): DE/rand/1/bin, population 20, F = 0.5, CR = 0.9.

**Random search** (the non-evolutionary baseline; Bergstra & Bengio 2012): 5000 genomes drawn
uniformly at random, no selection and no variation. Every evaluation goes into a map with the same
cells and the same rule as MAP-Elites (keep the better scheme per cell), so random search is
"MAP-Elites without evolution". With the same run seed, its first 100 genomes are exactly
MAP-Elites' initial random genomes; afterwards MAP-Elites mutates its elites while random search
keeps drawing new genomes.

**Fair comparison.** Same genome, same bounds, same fitness and the same budget of 5000 fitness
evaluations for all three algorithms; 10 independent runs each.

**Re-evaluation.** Picking the best of thousands of noisy 5-seed scores also picks lucky ones, so
the winner of every run and every baseline is retrained from scratch on 30 fresh seeds
(1000-1029) on:

| Maze | Role |
| --- | --- |
| `dyna_9x6_3traps` | training maze: sample efficiency |
| `dyna_18x12_3traps`, `dyna_27x18_3traps` | same layout, 2x and 3x bigger: effect of size only |
| `dyna_9x6_2traps` | same layout, different traps |
| `maze_8x8`, `maze_10x10`, `lava_7x5` | unseen layouts: generalisation |
| `unseen_6x8`, `unseen_7x7`, `unseen_9x7`, `unseen_11x8`, `unseen_12x10` | unseen random layouts (never used for training, also not by `multimaze/`) |

The `valid_*` mazes are not in this list on purpose: they are only for choosing configurations in
`screening/`, so no result is reported on the mazes a configuration was chosen with.

`analyze.py` reports the six main baselines (A1, A2, B1, B2, `sparse_dyna`,
`russell_norvig_matched`); `python analyze.py --all` includes every baseline and maze.

**Statistics.** The algorithms are compared pairwise (MAP-Elites vs DE, and each vs random search)
with one value per run (the winner's mean N90 over the fresh seeds). Against the baselines, every
winner is tested on its own
(30 vs 30 fresh seeds): the winners of different runs are different reward schemes, so pooling
them would overstate the sample size. `stats_summary.csv` counts how many winners are
significantly faster or slower than each baseline; `stats.csv` holds every test. All p-values are
Holm-corrected together.

**Quality-diversity.** `qd.csv` compares what each algorithm returns on the 10x10 grid: the
MAP-Elites archive, the random-search map and DE's final population. It reports cells filled, cells whose elite
learned on all 5 evaluation seeds, and the QD-score (sum over filled cells of 1000 - fitness).

## Saved results

`results/` is ignored by git, except for dated snapshots of complete experiments:

| Folder | Contents |
| --- | --- |
| `results/2026-11-15_laptop/` | 10 MAP-Elites + 10 DE runs (budget 5000), re-evaluation on 30 fresh seeds, analysis |

To redo the analysis of a snapshot: `python analyze.py --dir results/2026-11-15_laptop`.
To save a new experiment, move `results/evolution`, `results/reevaluation.csv` and
`results/analysis` into a new dated folder and add it to `.gitignore` like the existing one.


## First findings

- The 9x6 Dyna Maze with 2-3 traps gives the widest spread between reward schemes; without traps
  every scheme learns quickly, while in the 8x8 and 10x10 several schemes never learn.
- Delayed rewards fail once traps end the episode: progress collected before falling into a trap is
  paid out at the trap step, so falling into a trap after making progress looks rewarding.
- The progress reward uses shaping gamma = 1. With the agent's gamma (0.95) in the potential-based
  formula, standing still far from the goal earned a small reward every step and the agent learned
  to sit still.
- The first test above 90% is a noisy measure of learning: the greedy policy oscillates, so we
  use the moving average of the last 3 tests (see Agent and metrics).

## Next steps (plan, 23 Nov 2026)

Direction agreed after the TA's feedback ("something extra is missing"; bigger mazes are much
harder). Not implemented yet; the numbers below come from quick pilot checks.

### 1. How far does a reward scheme from the small maze scale?

- Train the evolved schemes (the winners and the whole MAP-Elites archive) on bigger copies of
  the training maze (`dyna_18x12_3traps`, `dyna_27x18_3traps` from the `scaled-mazes` branch,
  maybe a 5x copy) and find the size at which each scheme stops working.
- Which genes decide it, and which kinds of schemes (map cells) stay useful: a "transfer map",
  i.e. the MAP-Elites map coloured by performance on the bigger maze.
- A rule from the RL lecture (Russell & Norvig 4x3 world, optimal policy vs step reward): if
  walking to the goal pays less than walking into the nearest trap, the agent learns to end the
  episode in a trap. Step costs add up over a longer path, so this can flip when the maze grows;
  a progress reward can make up for it. The rule gives a predicted size limit for every scheme,
  to compare with the observed one.
- Optional reference: evolve directly on one big maze (server) and compare small-maze winners
  with big-maze winners on the big maze.
- Random search with the same budget as the non-evolutionary baseline.

Pilot (MAP-Elites run 7 archive and all 20 winners, 3 seeds, 1500 episodes on the 3x maze): all 20
winners still learn the 3x maze, but only 29 of the 54 learning elites do. The failures are mostly
long-delay and almost-only-punishment schemes. For 12 elites the trap pays more than the goal on
the 3x maze, and all 12 fail. On the 3x maze the hand-made B1 is as fast as the transferred
winners (180 vs 205-215 episodes), while most other hand-made schemes fail or slow down a lot.

### 2. Drift (slip) as a second condition

Our agent: with probability 0.1 the move is replaced by a random one of the 4 directions
(92.5% intended, 2.5% each side, 2.5% back). The RL lecture's 4x3 world uses 80% intended and
10% to each side.

- Test condition: re-test the schemes at higher drift; together with size this gives a
  "size x drift" limit for every scheme.
- Training drift as a gene (always tested at the normal drift). With learning speed as the only
  fitness, evolution would simply pick no drift, so this needs a second objective: safety (final
  success, trap falls). Multi-objective evolution (NSGA-II, lab 2) or a map axis.

Pilot (20 seeds on the small maze, 6 on the 3x maze, always tested at drift 0.1): more drift in
training means slower learning (DE winner on the small maze: 100 / 120 / 150 / 195 episodes at
drift 0 / 0.1 / 0.2 / 0.3) but slightly safer final policies (final success about 0.90 at drift 0
vs 0.94 at 0.2; B1 on the 3x maze 0.92 vs 0.99). Drift only in the first 30% of training
("early 0.3") was not faster. Drift interacts with the scheme: on the small maze B1 slows down
much more with drift than the evolved winners (100 -> 205 vs 100 -> 100-120 episodes).

### Implementation steps

- [x] 1. Random search as a third algorithm: `evolve.random_search`, `run_evolution.py --algo random`,
  label "RS best" in `reevaluate.py` and `analyze.py`. Same budget and output format as DE.
- [x] 2. Multi-maze evolution (`multimaze/`, see its README): the fitness of a scheme is its average
  over a fixed, seeded set of small random training mazes of different sizes (about 6x6 to 12x8,
  with traps), 1 seed per maze, so one evaluation costs about the same as now. Each maze's result
  is divided by that maze's episode budget before averaging. The test mazes (the hand-made ones
  and the scaled copies) are never used for training. The existing runs become the
  "single-maze" condition. Question: do schemes evolved on several small mazes learn bigger and
  unseen mazes faster than single-maze schemes and hand-made ones?
- [x] 3. `scaling/transfer.py` (see `scaling/README.md`): train new agents with the saved reward schemes (the winners of every
  run, every MAP-Elites elite, and the hand-made baselines; from single-maze, multi-maze and
  random-search runs) on bigger copies of the training maze, with new seeds. One CSV with
  episodes to learn, learned or not, final success and trap falls. Adds a 4x (and maybe 5x) copy
  to `gridmaze.SCALED` (added: 4x and 5x).
- [x] Screening (`screening/`, see its README): every configuration (training set, `--combine`,
  gene ranges) gets 3 short runs (budget 2000) for MAP-Elites and DE; the winners are compared on
  five separate validation mazes (`valid_*`), so the test mazes stay untouched. Full runs only for
  the one or two best. Needed for it: `--combine mean|relative|worst` and `--variant` in
  `multimaze/run_multimaze.py`.
- [ ] 4. `scaling/rule.py`: for every scheme and maze size, the total (discounted) reward of walking
  to the goal vs walking into the nearest trap, and the predicted size limit. Compared with the
  observed size limit from step 3.
- [ ] 5. Figures: a transfer map (the MAP-Elites map coloured by each scheme's size limit, with the
  predicted failures marked) and a size curve (episodes to learn vs maze size, evolved winners vs
  hand-made baselines).
- [ ] 6. Drift as a test condition: a drift-style option in `gridmaze.py` ("uniform", ours and the
  default; "sideways", the lecture's 80% / 10% / 10%), separate training and test drift in
  `qlearning.train()`, and `--drift` in `transfer.py`.

Optional:

- Intelligent trial and error (Cully et al. 2015) on the big maze: use the MAP-Elites map's
  small-maze scores as a first guess, train with the most promising scheme, update the guesses
  for the neighbouring cells, and stop when one is good enough. A few trainings instead of a new
  evolution. In the pilot, the best elite of the map needed 200 episodes on the 3x maze, the
  small-maze winner 290 and the hand-made B1 180.

Later:

- Multi-maze evolution with a 2x maze in the training set (about 2.5x the cost; server), to
  compare with step 2.
- Evolve the training drift together with the reward scheme (two objectives: speed and safety,
  e.g. NSGA-II), as a variant.
- Evolve directly on one big maze as a reference for the transferred schemes. Needs `--episodes` in
  `run_evolution.py` and a "never learned" penalty of 2 x episodes in `evolve.py`; server.
- SLURM script for the server (update `run_all.slurm` from `new-branch`) once we have access.

### Open decisions

- Multi-maze training set: chosen for now (see `multimaze/README.md`): the training maze plus five
  random mazes from 6x6 to 12x8 with 2 to 6 traps, one run each, plain mean of N90. Screening
  compares it with 2 runs per maze (`A2`), the `relative` and `worst` combinations, and a wider
  progress range; set B (with the 2x maze) later on the server.
- Screening protocol: chosen for now (see `screening/README.md`, with alternatives and how to
  switch): 3 runs per algorithm, budget 2000, MAP-Elites and DE, winners re-evaluated on the 5
  validation mazes with 10 seeds, scored by the mean episodes to 90%, `single` as the reference.
- Transfer test (step 3): chosen for now (see `scaling/README.md`, with alternatives and how to
  switch): sizes 1x to 4x (5x available), 500 x size training episodes, "learns a size" = at
  least 80% of 10 seeds, size limit = biggest size learned with every smaller size learned too,
  all runs' winners plus the learning elites. To review after the first full run.
- Drift style for the new experiments: keep ours (a random one of the 4 directions) or switch to
  the lecture's (80% intended, 10% to each side).
- Whether evolving the training drift (two objectives) is in scope.

### Related work (for the report)

The PDFs are in the shared `papers-for-project` folder.

- Singh, Lewis, Barto & Sorg (2010), Intrinsically Motivated RL: An Evolutionary Perspective:
  rewards as designed by evolution; the best reward depends on the environments an agent faces.
- Niekum (2010), Evolved Intrinsic Reward Functions for RL: rewards evolved with genetic
  programming beat Singh et al.'s brute-force search; fitness over a set of environments.
- Kanagawa & Doya (2024), Evolution of Rewards for Food and Motor Action by Simulating Birth and
  Death: evolved rewards change with environment size; in a small, easy environment the
  selection pressure is weak and the rewards hardly evolve.
- Zheng et al. (2020), What Can Learned Intrinsic Rewards Capture?: learned rewards transfer to
  other agents and to changed dynamics, transferred policies do not. Why we transfer the reward
  scheme and train a new agent.
- Chiang, Faust, Fiser & Francis (2019), Learning Navigation Behaviors End-to-End with AutoRL:
  evolutionary reward search in a small training map, policies used in buildings up to 30x
  bigger (they transfer the policy, we transfer the reward). Notes agents ending episodes early
  to exploit the reward.
- Cully, Clune, Tarapore & Mouret (2015), Robots that can adapt like animals: the MAP-Elites map
  as a first guess when conditions change (intelligent trial and error).
- Usaratniwart, Gao, Ong & Akimoto (2026), Evolutionary Bilevel Reward Shaping for Generalization
  in RL (GERS): CMA-ES tunes reward weights using scores on other environments than the training
  one; plain PPO overfits its training environment.
- Taylor & Stone (JMLR 2007 or their 2009 survey): transfer between smaller and bigger versions of
  a task; time to reach a threshold and total time (source + target) as transfer measures.
- One line each: Oh et al. (2020), Discovering RL Algorithms (an update rule found on toy grid
  worlds works on Atari); Goyal et al. (2019), InfoBot (random exploration reaches the goal less
  and less often as environments grow); Bewley et al. (2022), Reward Learning with Trees
  (negative rewards on every step push agents to end episodes; next to the Russell & Norvig
  slide for the trap rule).

### Ideas for later

- Co-evolution of worlds and reward schemes (Lecture 11, POET): mazes (traps, drift, size) evolve
  to be hard for the current reward schemes, and the schemes evolve to cope with them.
- Compression of big mazes (fixed blocks or a quadtree, with a danger margin around traps):
  evolve or plan on the small version and use it on the big one. Prototype: guiding the progress
  reward with the distance on a compressed map works on 30x30 and 50x50 mazes (slower than with
  the true distance, while without any guidance the agent never learns).


## References

- Mouret & Clune (2015), Illuminating search spaces by mapping elites (MAP-Elites)
- Auerbach, Iacca & Floreano (2016), Gaining Insight into Quality Diversity
- Singh, Lewis & Barto (2009), Where Do Rewards Come From?
- Niekum, Barto & Spector (2010), Genetic Programming for Reward Function Search
- Sutton & Barto (2018), Reinforcement Learning: An Introduction (Dyna Maze, Example 8.1)
- Leike et al. (2017), AI Safety Gridworlds
- Ng, Harada & Russell (1999), Policy Invariance Under Reward Transformations
- Arjona-Medina et al. (2019), RUDDER: Return Decomposition for Delayed Rewards
- Storn & Price (1997), Differential Evolution: A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces
- Mann & Whitney (1947), On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other
- Holm (1979), A Simple Sequentially Rejective Multiple Test Procedure
- Vargha & Delaney (2000), A Critique and Improvement of the CL Common Language Effect Size Statistics of McGraw and Wong
