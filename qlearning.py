"""Tabular Q-learning agent trained under a given reward scheme.

The agent settings are fixed for every reward scheme, so differences in learning
come from the rewards only.
"""
from collections import deque
import math
import random

DEFAULTS = dict(
    episodes=500,           # training episodes per run
    alpha=0.2,              # learning rate
    gamma=0.95,             # discount
    eps_start=1.0,          # epsilon-greedy exploration: start ...
    eps_end=0.05,           # ... end ...
    eps_decay_frac=0.7,     # ... reached after this fraction of the episodes (linear decay)
    time_limit_factor=10,   # episode time limit = factor x shortest path
    eval_every=10,          # test the greedy policy every N episodes ...
    eval_episodes=20,       # ... with this many test episodes
    success_threshold=0.9,  # "learned" = test success rate reaches this
    shaping_gamma=1.0,      # progress reward = shaping_gamma * Phi(s') - Phi(s).
                            # 1.0 gives +w per step closer, -w per step away, 0 for standing still.
                            # Using the agent's gamma (0.95) pays +w*d*(1-gamma) for standing still
                            # far from the goal, which made the agent learn to sit still in our tests.
)


def _greedy(q_s, rng):
    m = max(q_s)
    best = [a for a in range(4) if q_s[a] == m]
    return best[0] if len(best) == 1 else rng.choice(best)


def evaluate(maze, Q, time_limit, n_episodes, rng):
    """Success rate and mean path length of the greedy policy (slip still applies)."""
    successes, lengths = 0, []
    for _ in range(n_episodes):
        s = maze.start
        for t in range(time_limit):
            s, _, goal, trap = maze.step(s, _greedy(Q[s], rng), rng)
            if goal:
                successes += 1
                lengths.append(t + 1)
                break
            if trap:
                break
    return successes / n_episodes, (sum(lengths) / len(lengths) if lengths else math.nan)


def train(maze, scheme, seed, **settings):
    """Train one agent from scratch. Returns a dict of results for this run."""
    cfg = {**DEFAULTS, **settings}
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 1_000_003)
    alpha, gamma = cfg["alpha"], cfg["gamma"]
    T = cfg["time_limit_factor"] * maze.shortest_path
    n_ep = cfg["episodes"]
    phi = [-scheme.progress * d for d in maze.dist]     # potential for the progress reward
    delay = scheme.delay

    Q = [[0.0] * 4 for _ in range(maze.n_states)]
    curve, eval_points = [], []
    env_steps = 0
    first_90_episode = first_90_steps = None
    trap_falls = wall_bumps = 0
    pos_feedback = neg_feedback = 0.0                  # for the punishment share
    delay_weighted = delay_weight = 0.0                # for the mean feedback delay

    for ep in range(n_ep):
        eps = max(cfg["eps_end"],
                  cfg["eps_start"] - (cfg["eps_start"] - cfg["eps_end"]) * ep / (cfg["eps_decay_frac"] * n_ep))
        s = maze.start
        pending = deque()                              # (due_step, value, created_step)
        for t in range(T):
            a = rng.randrange(4) if rng.random() < eps else _greedy(Q[s], rng)
            ns, bump, goal, trap = maze.step(s, a, rng)
            env_steps += 1
            done = goal or trap

            # rewards of this step, component by component
            parts = []
            if goal:
                parts.append(scheme.goal * (1 - 0.9 * (t + 1) / T) if scheme.minigrid_goal else scheme.goal)
            if trap:
                parts.append(scheme.trap)
                trap_falls += 1
            if bump:
                parts.append(scheme.wall)
                wall_bumps += 1
            if scheme.step:
                parts.append(scheme.step)
            if scheme.progress:
                parts.append(cfg["shaping_gamma"] * phi[ns] - phi[s])
            for v in parts:
                if v > 0:
                    pos_feedback += v
                else:
                    neg_feedback -= v
            r = sum(parts)

            # timing: when does the agent receive it?
            last = done or t == T - 1
            if delay == 0:
                given = r
                if r:
                    delay_weight += abs(r)
            else:
                if r:
                    due = math.inf if delay == "end" else t + delay
                    pending.append((due, r, t))
                given = 0.0
                while pending and (last or pending[0][0] <= t):
                    _, v, created = pending.popleft()
                    given += v
                    delay_weighted += abs(v) * (t - created)
                    delay_weight += abs(v)

            # Q-learning update
            target = given if done else given + gamma * max(Q[ns])
            Q[s][a] += alpha * (target - Q[s][a])
            s = ns
            if done:
                break

        if (ep + 1) % cfg["eval_every"] == 0:
            rate, _ = evaluate(maze, Q, T, cfg["eval_episodes"], eval_rng)
            curve.append(rate)
            eval_points.append(ep + 1)
            if first_90_episode is None and rate >= cfg["success_threshold"]:
                first_90_episode, first_90_steps = ep + 1, env_steps

    final_rate, final_len = evaluate(maze, Q, T, 100, eval_rng)
    total_feedback = pos_feedback + neg_feedback
    return dict(
        maze=maze.name,
        scheme=scheme.name,
        source=scheme.source,
        seed=seed,
        shortest_path=maze.shortest_path,
        episodes_to_90=first_90_episode,
        steps_to_90=first_90_steps,
        final_success=final_rate,
        final_path_length=final_len,
        trap_falls=trap_falls,
        wall_bumps=wall_bumps,
        punishment_share=neg_feedback / total_feedback if total_feedback else 0.0,
        mean_feedback_delay=delay_weighted / delay_weight if delay_weight else 0.0,
        curve=curve,
        eval_points=eval_points,
    )
