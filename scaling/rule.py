"""Step 4: does a reward scheme make the goal the best way to end an episode?

The lecture's rule (Russell & Norvig 4x3 world): with a step cost, a long path to the goal can be
worth less than a short walk into a trap, and then the best policy ends in the trap. Step costs
add up with the path length, so a scheme that is fine on the small maze can flip on a bigger one.

This script checks that for every scheme and maze size, from the rewards alone (no learning):
it finds the path with the highest return the agent can get, and looks at where it ends.

Method (method 2 in scaling/README.md): an exact search over all paths in the maze WITHOUT slip
(deterministic moves), with the same rewards, discount (gamma = 0.95), time limit (10 x shortest
path) and reward timing as qlearning.train():
    - goal, trap, wall, step and progress rewards exactly as in train()
    - delayed rewards (delay k): a reward earned at step t is paid at step t + k, and whatever is
      still pending when the episode ends is paid at the last step; "end" pays everything at the
      last step. So the goal or trap reward itself is never delayed.
    - three ways an episode can end: at the goal, in a trap, or at the time limit (wandering)
Result per scheme and size: the best return for each ending, the best ending, and the margin
(best goal return minus the best other one). The predicted size limit is the biggest size where
the goal is the best ending, with every smaller size too (as the observed size limit in
transfer.py).

How the delays are handled exactly: for a path that ends after tau steps, a reward earned j steps
before the end is paid at step tau - 1 - max(j - k, 0). So the return is
    gamma^(tau-1) * (R_end + sum over steps of gamma^(-max(j-k, 0)) * r_step)
and the sum is maximised by dynamic programming over j (steps left), for every tau at once.

What this does NOT include (see scaling/README.md for the other two methods): the 10% slip
(method 3 would use the same search with random moves), and how fast the agent can find the
best path. So "goal is best" is necessary for learning the maze, not enough: a scheme can have the
right incentive and still be too slow. The comparison with transfer.py separates the two.

Run from the repository folder:
    python scaling/rule.py                                    # saved single-maze runs, 1x-5x
    python scaling/rule.py --transfer results/scaling/transfer/transfer.csv     # + compare
    python scaling/rule.py --dirs results/2026-11-15_laptop results/multimaze/A --name single_vs_multi

Output: results/scaling/<name>/rule.csv (one row per scheme and size) and rule_summary.csv (one
row per scheme: predicted size limit, and the observed one if --transfer is given).
"""
import argparse
import csv
import math
import os
import statistics
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from qlearning import DEFAULTS  # noqa: E402
from rewards import baseline_schemes  # noqa: E402
import transfer  # noqa: E402  (same candidates, keys and maze list as the transfer test)

ENDINGS = ("goal", "trap", "timeout")


def maze_arrays(maze, shaping_gamma=DEFAULTS["shaping_gamma"]):
    """Deterministic moves as arrays: next state, bump, progress change, kind of the next cell."""
    nxt = np.array(maze.next_state, dtype=np.int64)                      # (S, 4)
    S = maze.n_states
    bump = (nxt == np.arange(S)[:, None]).astype(float)
    d = np.array(maze.dist, dtype=float)
    progress = d[:, None] - shaping_gamma * d[nxt]                        # times w: the progress reward
    is_goal = nxt == maze.goal
    is_trap = np.array(maze.is_trap)[nxt]
    return nxt, bump, progress, is_goal, is_trap


def best_returns(maze, schemes, gamma=DEFAULTS["gamma"], time_limit=None, chunk=256):
    """For every scheme: best return of each ending, and the length of the best path of each.

    Returns a list of dicts (one per scheme): ret_goal, ret_trap, ret_timeout, len_goal, len_trap.
    -inf = this ending is impossible (e.g. no trap in the maze)."""
    T = time_limit or DEFAULTS["time_limit_factor"] * maze.shortest_path
    nxt, bump, progress, is_goal, is_trap = maze_arrays(maze)
    ongoing = ~(is_goal | is_trap)
    s0 = maze.start
    out = []
    for c0 in range(0, len(schemes), chunk):
        sc = schemes[c0:c0 + chunk]
        n = len(sc)
        col = lambda f: np.array([f(s) for s in sc], dtype=float)[:, None, None]  # noqa: E731
        # reward of every move except the goal/trap reward itself, as in qlearning.train()
        r = col(lambda s: s.wall) * bump + col(lambda s: s.step) + col(lambda s: s.progress) * progress
        k = np.array([math.inf if s.delay == "end" else float(s.delay) for s in sc])
        goal = np.array([s.goal for s in sc])
        trap = np.array([s.trap for s in sc])
        minigrid = np.array([s.minigrid_goal for s in sc])

        F = {}                                         # F[end][i, s]: best weighted sum, j steps left
        F["goal"] = np.where(is_goal, r, -np.inf).max(axis=2)
        F["trap"] = np.where(is_trap, r, -np.inf).max(axis=2)
        F["timeout"] = np.where(ongoing, r, -np.inf).max(axis=2)
        best = {e: np.full(n, -np.inf) for e in ENDINGS}
        length = {e: np.zeros(n, dtype=int) for e in ENDINGS}
        for j in range(T):                             # j = tau - 1: steps after the first one
            tau = j + 1
            disc = gamma ** j
            if j > 0:
                w = gamma ** (-np.maximum(j - k, 0.0))[:, None, None]       # weight of the new first step
                for e in ENDINGS:
                    F[e] = np.where(ongoing, w * r + F[e][:, nxt], -np.inf).max(axis=2)
            R = np.where(minigrid, goal * (1 - 0.9 * tau / T), goal)
            for e, end_reward in (("goal", R), ("trap", trap)):
                v = disc * (F[e][:, s0] + end_reward)
                better = v > best[e]
                best[e] = np.where(better, v, best[e])
                length[e] = np.where(better, tau, length[e])
        best["timeout"] = gamma ** (T - 1) * F["timeout"][:, s0]
        length["timeout"][:] = T
        for i in range(n):
            out.append({f"ret_{e}": float(best[e][i]) for e in ENDINGS}
                       | {"len_goal": int(length["goal"][i]), "len_trap": int(length["trap"][i])})
    return out


def judge(res, tol=1e-9):
    """Best ending and the margin of the goal over the best other ending."""
    other = max(res["ret_trap"], res["ret_timeout"])
    ending = max(ENDINGS, key=lambda e: res[f"ret_{e}"])
    if res["ret_goal"] >= other - tol:
        ending = "goal"
    return ending, res["ret_goal"] - other


def size_limit(ok_by_scale):
    limit = 0
    for k in sorted(ok_by_scale):
        if not ok_by_scale[k]:
            break
        limit = k
    return limit


def observed_limits(transfer_csv, threshold):
    """candidate key -> {scale: share of seeds that learned}, at the normal drift (uniform 0.1)"""
    by = defaultdict(lambda: defaultdict(list))
    normal = transfer.drift_label(*transfer.NORMAL_DRIFT)
    for r in csv.DictReader(open(transfer_csv)):
        if transfer._condition(r) != normal:      # the rule has no drift: compare with the normal one
            continue
        by[r["candidate"]][int(r["scale"])].append(r["learned"] == "True")
    return {key: {k: sum(v) / len(v) for k, v in sc.items()} for key, sc in by.items()}


def compare(summary, rows, obs, scales, threshold):
    """Print how well the prediction matches what the agents did in transfer.py."""
    print("\n=== prediction vs transfer test ===")
    print("predicted: the goal is the best ending (deterministic maze); observed: >= "
          f"{threshold:.0%} of the seeds reached 90% success\n")
    print(f"{'size':>5s}{'schemes':>9s}{'goal best':>11s}{'  of these learned':>20s}{'goal not best':>15s}{'  of these learned':>20s}")
    for k in scales:
        rs = [r for r in rows if r["scale"] == k and r["candidate"] in obs and k in obs[r["candidate"]]]
        good = [r for r in rs if r["best_ending"] == "goal"]
        bad = [r for r in rs if r["best_ending"] != "goal"]
        learned = lambda group: sum(obs[r["candidate"]][k] >= threshold for r in group)  # noqa: E731
        print(f"{k:>4d}x{len(rs):9d}{len(good):11d}{learned(good):20d}{len(bad):15d}{learned(bad):20d}")
    print("\nwhere the goal is the best ending: schemes that learned vs not (medians; evolved schemes only;")
    print("delay 'end' counted as 999)")
    num = lambda v: 999.0 if v == "end" else float(v)  # noqa: E731
    med = lambda grp, f: f"{statistics.median(num(r[f]) for r in grp):9.2f}" if grp else f"{'-':>9s}"  # noqa: E731
    print(f"{'size':>5s} {'':12s}{'schemes':>8s}{'margin':>9s}" + "".join(f"{g:>9s}" for g in transfer.GENES))
    for k in scales:
        rs = [r for r in rows if r["scale"] == k and r["best_ending"] == "goal" and r["kind"] != "baseline"
              and k in obs.get(r["candidate"], {})]
        for label, grp in (("learned", [r for r in rs if obs[r["candidate"]][k] >= threshold]),
                           ("not learned", [r for r in rs if obs[r["candidate"]][k] < threshold])):
            print(f"{k:>4d}x {label:12s}{len(grp):8d}{med(grp, 'margin')}" + "".join(med(grp, g) for g in transfer.GENES))
    both = [s for s in summary if s["observed_limit"] != ""]
    if not both:
        return
    same = sum(s["observed_limit"] == s["predicted_limit"] for s in both)
    above = sum(s["observed_limit"] < s["predicted_limit"] for s in both)
    below = sum(s["observed_limit"] > s["predicted_limit"] for s in both)
    print(f"\nsize limit, {len(both)} schemes: predicted = observed {same}, "
          f"observed lower (too slow although the goal is best) {above}, "
          f"observed higher (learned although the goal is not best) {below}")
    print("observed lower: the incentive is right, the agent is too slow (exploration, episodes).")
    print("observed higher: the rule is wrong for this scheme (slip, delays, or the agent never finds the")
    print("better ending, e.g. a trap it rarely reaches); worth a look.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", default=[os.path.join(ROOT, "results", "2026-11-15_laptop")])
    p.add_argument("--runs", nargs="+", type=int, help="only these run numbers (default: all)")
    p.add_argument("--elites", choices=["learning", "all", "none"], default="learning")
    p.add_argument("--baselines", choices=["main", "all", "none"], default="all")
    p.add_argument("--scales", nargs="+", type=int, default=sorted(transfer.SCALED), choices=sorted(transfer.SCALED))
    p.add_argument("--transfer", help="a transfer.csv to compare with (same --dirs/--runs as that run)")
    p.add_argument("--threshold", type=float, default=0.8, help="'learns a size' as in transfer.py")
    p.add_argument("--name", default="rule", help="output folder results/scaling/<name>")
    a = p.parse_args()

    cands = transfer.load_candidates(a.dirs, a.elites, a.runs)
    print(f"{len(cands)} evolved schemes from {len(a.dirs)} folder(s) + baselines, sizes {a.scales}")
    rows, t0 = [], time.time()
    for k in a.scales:
        maze = transfer._maze(transfer.SCALED[k])
        here = list(cands)
        if a.baselines != "none":
            for s in baseline_schemes(maze):
                if a.baselines == "all" or any(s.name.startswith(b) for b in transfer.MAIN_BASELINES):
                    here.append(dict(source="hand-made", condition="hand-made", algo="", run="",
                                     kind="baseline", cell="", train_fitness="", scheme=s))
        results = best_returns(maze, [c["scheme"] for c in here])
        for c, res in zip(here, results):
            ending, margin = judge(res)
            rows.append(dict(candidate=transfer.candidate_key(c), **transfer.cand_fields(c), scale=k,
                             maze=maze.name, shortest_path=maze.shortest_path, best_ending=ending,
                             margin=round(margin, 4), **{x: round(v, 4) for x, v in res.items()}))
        print(f"  {k}x ({maze.name}, path {maze.shortest_path}): {len(here)} schemes, "
              f"goal best for {sum(r['best_ending'] == 'goal' for r in rows if r['scale'] == k)}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    obs = observed_limits(a.transfer, a.threshold) if a.transfer else {}
    per = defaultdict(dict)
    for r in rows:
        per[r["candidate"]][r["scale"]] = r
    summary = []
    for key, by_scale in per.items():
        first = next(iter(by_scale.values()))
        s = {f: first[f] for f in ("candidate", "source", "condition", "kind", "algo", "run", "cell", "scheme",
                                   *transfer.GENES)}
        for k in a.scales:
            if k in by_scale:
                s[f"ending_{k}x"] = by_scale[k]["best_ending"]
                s[f"margin_{k}x"] = by_scale[k]["margin"]
        s["predicted_limit"] = size_limit({k: r["best_ending"] == "goal" for k, r in by_scale.items()})
        if key in obs:
            s["observed_limit"] = size_limit({k: v >= a.threshold for k, v in obs[key].items()})
        else:
            s["observed_limit"] = ""
        summary.append(s)

    out_dir = os.path.join(ROOT, "results", "scaling", a.name)
    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("rule.csv", rows), ("rule_summary.csv", summary)):
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(x for d in data for x in d)))
            w.writeheader()
            w.writerows(data)
    print(f"saved {os.path.join(out_dir, 'rule.csv')} and rule_summary.csv")

    # short tables: groups of evolved schemes, then every baseline
    print("\nshare of schemes for which the goal is the best ending, by size:")
    groups = defaultdict(list)
    for s in summary:
        g = "hand-made" if s["kind"] == "baseline" else f"{s['condition']}, {s['algo']} {s['kind']}s"
        groups[g].append(s)
    head = "".join(f"{f'{k}x':>7s}" for k in a.scales)
    print(f"{'group':44s}{'schemes':>8s}{head}{'limit (median)':>16s}")
    for g in sorted(groups, key=lambda g: (g == "hand-made", g)):
        ss = groups[g]
        shares = "".join(f"{sum(x.get(f'ending_{k}x') == 'goal' for x in ss) / len(ss):7.0%}" for k in a.scales)
        print(f"{g:44s}{len(ss):8d}{shares}{statistics.median(x['predicted_limit'] for x in ss):16.0f}")
    print("\nhand-made schemes: best ending per size (margin = best goal return - best other return)")
    for s in sorted((s for s in summary if s["kind"] == "baseline"), key=lambda s: s["scheme"]):
        cells = "".join(f"{s.get(f'ending_{k}x', ''):>9s} {s.get(f'margin_{k}x', 0):+6.2f}" for k in a.scales)
        print(f"  {s['scheme']:28s}{cells}")
    if obs:
        compare(summary, rows, obs, a.scales, a.threshold)


if __name__ == "__main__":
    main()
