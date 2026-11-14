"""Evolving reward schemes (this is a LIBRARY: running it directly prints nothing;
use run_evolution.py): genome, fitness, MAP-Elites and Differential Evolution.

Both algorithms search the SAME space with the SAME fitness and the SAME budget
(number of fitness evaluations), so their results can be compared directly.

Genome (5 genes, each stored normalised in [0, 1]):
    trap, wall, step, progress, delay
The goal reward is fixed at +10. For tabular Q-learning starting from all-zero
Q-values, multiplying every reward by the same positive number changes nothing,
so only the ratios to the goal matter and a free goal gene would be redundant.

Fitness (to minimise): mean episodes_to_90 over the evaluation seeds,
with PENALTY for every seed that never reaches 90% test success.

Behaviour descriptors (MAP-Elites axes), measured during training, averaged over seeds:
    punishment_share     share of all feedback that was negative       [0, 1]
    mean_feedback_delay  average steps between an event and its reward  [0, MAX_DELAY]
"""
import math
import random

from qlearning import DEFAULTS, train
from rewards import RewardScheme

GOAL = 10.0
MAX_DELAY = 30                      # largest delay in steps (about 2x the shortest path)
GENES = ["trap", "wall", "step", "progress", "delay"]
BOUNDS = {                          # real value = low + x * (high - low)
    "trap": (-20.0, 0.0),           # up to 2x the goal
    "wall": (-5.0, 0.0),
    "step": (-1.0, 0.0),            # -1 per step = shortest path costs more than the goal
    "progress": (0.0, 1.0),         # potential-based shaping weight
    "delay": (0, MAX_DELAY),        # integer number of steps; 0 = immediate
}
PENALTY = 2 * DEFAULTS["episodes"]  # fitness of a seed that never learns
EVAL_SEEDS = (0, 1, 2, 3, 4)        # seeds used DURING evolution (fresh ones for re-evaluation)
GRID = (10, 10)                     # MAP-Elites cells along (punishment_share, delay)


# ---------------------------------------------------------------------------
# Genome <-> reward scheme
# ---------------------------------------------------------------------------
def decode(x, name="evolved"):
    """Normalised genome (list of 5 floats in [0, 1]) -> RewardScheme."""
    v = {}
    for g, xi in zip(GENES, x):
        lo, hi = BOUNDS[g]
        v[g] = lo + min(1.0, max(0.0, xi)) * (hi - lo)
    return RewardScheme(name, goal=GOAL, trap=v["trap"], wall=v["wall"], step=v["step"],
                        progress=v["progress"], delay=int(round(v["delay"])))


def random_genome(rng):
    return [rng.random() for _ in GENES]


# ---------------------------------------------------------------------------
# Fitness + descriptors
# ---------------------------------------------------------------------------
def evaluate(x, maze, seeds=EVAL_SEEDS, **settings):
    """Train one agent per seed. Returns a dict with fitness and descriptors."""
    scheme = decode(x)
    runs = [train(maze, scheme, s, **settings) for s in seeds]
    n90 = [r["episodes_to_90"] if r["episodes_to_90"] is not None else PENALTY for r in runs]
    return dict(
        genome=list(x),
        fitness=sum(n90) / len(n90),
        n90=n90,
        converged=sum(r["episodes_to_90"] is not None for r in runs),
        punishment_share=sum(r["punishment_share"] for r in runs) / len(runs),
        mean_feedback_delay=sum(r["mean_feedback_delay"] for r in runs) / len(runs),
    )


def cell_of(ev):
    """Which MAP-Elites cell an evaluated genome falls into."""
    i = min(GRID[0] - 1, int(ev["punishment_share"] * GRID[0]))
    j = min(GRID[1] - 1, int(ev["mean_feedback_delay"] / MAX_DELAY * GRID[1]))
    return i, j


# ---------------------------------------------------------------------------
# MAP-Elites (Mouret & Clune 2015)
# ---------------------------------------------------------------------------
def map_elites(evaluate_batch, budget, seed, n_init=100, batch=50, sigma=0.1, log=None):
    """evaluate_batch(list_of_genomes) -> list of evaluation dicts (can run in parallel).

    1. Evaluate n_init random genomes and put each in its cell if it beats the current elite.
    2. Repeat: pick random elites, mutate them (Gaussian, sigma in normalised units),
       evaluate, and keep each child if its cell is empty or it is better than the elite.
    Returns (archive dict cell -> evaluation, history of best fitness per evaluation count).
    """
    rng = random.Random(seed)
    archive, history, used = {}, [], 0

    def insert(evs):
        nonlocal used
        for ev in evs:
            used += 1
            c = cell_of(ev)
            if c not in archive or ev["fitness"] < archive[c]["fitness"]:
                archive[c] = ev
            history.append((used, min(e["fitness"] for e in archive.values()), len(archive)))

    insert(evaluate_batch([random_genome(rng) for _ in range(min(n_init, budget))]))
    if log:
        log(f"MAP-Elites {used}/{budget}  best={history[-1][1]:.1f}  cells={len(archive)}  (initial random genomes)")
    while used < budget:
        n = min(batch, budget - used)
        elites = list(archive.values())
        children = []
        for _ in range(n):
            parent = rng.choice(elites)["genome"]
            children.append([min(1.0, max(0.0, g + rng.gauss(0, sigma))) for g in parent])
        insert(evaluate_batch(children))
        if log:
            log(f"MAP-Elites {used}/{budget}  best={history[-1][1]:.1f}  cells={len(archive)}")
    return archive, history


# ---------------------------------------------------------------------------
# Differential Evolution, DE/rand/1/bin (Storn & Price 1997)
# ---------------------------------------------------------------------------
def differential_evolution(evaluate_batch, budget, seed, pop_size=20, F=0.5, CR=0.9, log=None):
    """Classic DE. Every generation evaluates pop_size trial vectors in one batch.
    Returns (best evaluation, final population, history of best fitness per evaluation count).
    """
    rng = random.Random(seed)
    d = len(GENES)
    pop = evaluate_batch([random_genome(rng) for _ in range(pop_size)])
    used = len(pop)
    best = min(pop, key=lambda e: e["fitness"])
    history = [(i + 1, min(e["fitness"] for e in pop[: i + 1]), None) for i in range(len(pop))]
    if log:
        log(f"DE {used}/{budget}  best={best['fitness']:.1f}  (initial population)")
    while used < budget:
        trials = []
        n = min(pop_size, budget - used)
        for i in range(n):
            a, b, c = rng.sample([j for j in range(pop_size) if j != i], 3)
            xa, xb, xc = pop[a]["genome"], pop[b]["genome"], pop[c]["genome"]
            jrand = rng.randrange(d)
            trial = []
            for k in range(d):
                if rng.random() < CR or k == jrand:
                    trial.append(min(1.0, max(0.0, xa[k] + F * (xb[k] - xc[k]))))
                else:
                    trial.append(pop[i]["genome"][k])
            trials.append(trial)
        for i, ev in enumerate(evaluate_batch(trials)):
            used += 1
            if ev["fitness"] <= pop[i]["fitness"]:      # greedy one-to-one selection
                pop[i] = ev
            if ev["fitness"] < best["fitness"]:
                best = ev
            history.append((used, best["fitness"], None))
        if log:
            log(f"DE {used}/{budget}  best={best['fitness']:.1f}")
    return best, pop, history
