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
| `evolve.py` | Reward genome (5 genes, goal fixed at +10), fitness, behaviour descriptors, MAP-Elites and Differential Evolution |
| `run_evolution.py` | One evolutionary run (algorithm + run seed) from the command line, saved to `results/evolution/<algo>_run<N>.json` |
| `run_all.slurm` | SLURM job array: all 10 MAP-Elites + 10 DE runs in parallel on a cluster |
| `plot_archive.py` | Heatmap of a MAP-Elites archive (fitness or a gene value per cell) |
| `reevaluate.py` | Retrains the evolved winners and the baselines on 30 fresh seeds, on the training maze and on unseen mazes |
| `analyze.py` | Summary tables, Mann-Whitney U tests with Holm correction, Vargha-Delaney A12, boxplots, convergence curves |

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

**Fair comparison.** Same genome, same bounds, same fitness and the same budget of 5000 fitness
evaluations for both algorithms; 10 independent runs each.

**Re-evaluation.** Picking the best of thousands of noisy 5-seed scores also picks lucky ones, so
the winner of every run and every baseline is retrained from scratch on 30 fresh seeds
(1000-1029) on:

| Maze | Role |
| --- | --- |
| `dyna_9x6_3traps` | training maze: sample efficiency |
| `dyna_9x6_2traps` | same layout, different traps |
| `maze_8x8`, `maze_10x10`, `lava_7x5` | unseen layouts: generalisation |

`analyze.py` reports the six main baselines (A1, A2, B1, B2, `sparse_dyna`,
`russell_norvig_matched`); `python analyze.py --all` includes every baseline and maze.


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
