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
- [ ] Fix the final maze (candidate: Sutton & Barto's 9x6 Dyna Maze with 2-3 traps)
- [ ] MAP-Elites over reward schemes
- [ ] Comparison with the baselines, test on new mazes, report

## Quick start

Python 3.10+. The code uses only the standard library; matplotlib is needed for the plots.

```
pip install -r requirements.txt
python run_baselines.py --quick
```

`--quick` runs 2 mazes x 12 reward schemes x 3 seeds in a few seconds. More options:

```
python run_baselines.py                                   # 6 mazes x 12 schemes x 10 seeds (about a minute)
python run_baselines.py --mazes dyna_9x6_2traps dyna_9x6_3traps --seeds 30
python run_baselines.py --random 5 --random-size 9x6 --random-traps 3
python run_baselines.py --schemes A1_immediate_reward_only B1_immediate_reward_punish sparse_dyna
python run_baselines.py --help
```

## Files

| File | What it does |
| --- | --- |
| `gridmaze.py` | The maze, built from text (walls, traps, slipping, shortest paths); named layouts in `LAYOUTS` and `random_maze()` |
| `rewards.py` | `RewardScheme` (goal, trap, wall, step, progress, delay) and `baseline_schemes()` |
| `qlearning.py` | Tabular Q-learning (`train`) with fixed settings in `DEFAULTS`, and the metrics |
| `run_baselines.py` | Runs mazes x schemes x seeds in parallel, saves CSVs and plots to `results/` |

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
| `episodes_to_90`, `steps_to_90` | training episodes / environment steps until 90% test success |
| `final_success` | test success after training |
| `trap_falls`, `wall_bumps` | counted during training |
| `punishment_share` | share of all feedback received that was negative (planned MAP-Elites axis) |
| `mean_feedback_delay` | average steps between an event and its reward (planned MAP-Elites axis) |

Outputs in `results/`: `summary.csv`, `runs.csv`, `overview.png` (median episodes to 90% for every
maze and scheme), `curves_<maze>.png`, `mazes.png`.

## First findings

- The 9x6 Dyna Maze with 2-3 traps gives the widest spread between reward schemes; without traps
  every scheme learns quickly, while in the 8x8 and 10x10 several schemes never learn.
- Delayed rewards fail once traps end the episode: progress collected before falling into a trap is
  paid out at the trap step, so falling into a trap after making progress looks rewarding.
- The progress reward uses shaping gamma = 1. With the agent's gamma (0.95) in the potential-based
  formula, standing still far from the goal earned a small reward every step and the agent learned
  to sit still.

## References

- Mouret & Clune (2015), Illuminating search spaces by mapping elites (MAP-Elites)
- Auerbach, Iacca & Floreano (2016), Gaining Insight into Quality Diversity
- Singh, Lewis & Barto (2009), Where Do Rewards Come From?
- Niekum, Barto & Spector (2010), Genetic Programming for Reward Function Search
- Sutton & Barto (2018), Reinforcement Learning: An Introduction (Dyna Maze, Example 8.1)
- Leike et al. (2017), AI Safety Gridworlds
- Ng, Harada & Russell (1999), Policy Invariance Under Reward Transformations
- Arjona-Medina et al. (2019), RUDDER: Return Decomposition for Delayed Rewards
