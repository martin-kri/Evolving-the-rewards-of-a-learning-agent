"""Reward schemes: what the environment pays for each event, and when.

All schemes use goal = +10 so they can be compared directly. For tabular Q-learning
that starts from all-zero Q-values, only the ratios between the rewards matter.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardScheme:
    name: str
    goal: float = 10.0      # reaching the goal (ends the episode)
    trap: float = 0.0       # stepping into a trap (ends the episode)
    wall: float = 0.0       # bumping into a wall or the edge
    step: float = 0.0       # every step (a step cost if negative)
    progress: float = 0.0   # weight w of potential-based shaping, Phi(s) = -w * distance_to_goal(s)
    delay: object = 0       # 0 = immediate; k (int) = every reward arrives k steps later;
                            # "end" = all rewards summed and paid at the last step of the episode
    minigrid_goal: bool = False  # goal pays goal * (1 - 0.9 * t / time_limit), as in MiniGrid
    source: str = ""


# Published step costs, rescaled to goal = +10, with the shortest path of the original maze
RUSSELL_NORVIG = dict(step=-0.04 * 10, path=5)      # 4x3 world: -0.04 per step, goal +1
AI_SAFETY_LAVA = dict(step=-1 * 10 / 50, path=8)    # lava world: -1 per step, goal +50


def size_matched(step_cost, orig_path, our_path):
    """Scale a per-step cost so the shortest path costs the same share of the goal."""
    return step_cost * orig_path / our_path


def baseline_schemes(maze, progress_weight=0.1):
    """All baseline schemes for one maze (size-matched step costs depend on its shortest path)."""
    L = maze.shortest_path
    w = progress_weight
    return [
        # our 2x2 hand-made baseline
        RewardScheme("A1_immediate_reward_only", progress=w, source="ours"),
        RewardScheme("A2_delayed_reward_only", progress=w, delay="end", source="ours"),
        RewardScheme("B1_immediate_reward_punish", trap=-5, wall=-1, progress=w, source="ours"),
        RewardScheme("B2_delayed_reward_punish", trap=-5, wall=-1, progress=w, delay="end", source="ours"),
        RewardScheme("B1_trap10", trap=-10, wall=-1, progress=w, source="ours, trap = -goal"),
        # published schemes, goal scaled to +10
        RewardScheme("sparse_dyna", source="Sutton & Barto Dyna Maze / FrozenLake"),
        RewardScheme("kenton", trap=-10, source="Kenton et al. 2019"),
        RewardScheme("russell_norvig", trap=-10, step=RUSSELL_NORVIG["step"],
                     source="Russell & Norvig, as published"),
        RewardScheme("russell_norvig_matched", trap=-10,
                     step=size_matched(RUSSELL_NORVIG["step"], RUSSELL_NORVIG["path"], L),
                     source="Russell & Norvig, size-matched"),
        RewardScheme("ai_safety", trap=-10, step=AI_SAFETY_LAVA["step"],
                     source="AI Safety lava world, as published"),
        RewardScheme("ai_safety_matched", trap=-10,
                     step=size_matched(AI_SAFETY_LAVA["step"], AI_SAFETY_LAVA["path"], L),
                     source="AI Safety lava world, size-matched"),
        RewardScheme("minigrid", minigrid_goal=True, source="MiniGrid"),
    ]
