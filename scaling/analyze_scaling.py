"""Scaling thesis: does the trap-vs-goal rule predict which reward schemes stop working on bigger
mazes, and why do the others fail?

Reads the output of transfer.py (what the agents did) and rule.py (what the rewards alone
predict), joined on the candidate key and the maze size, and produces:

  1. confusion.csv      per size: rule prediction (goal best or not) x observed (learned or not).
                        "fail" predictions that came true, and how many observed failures the rule
                        explains.
  2. limits.csv         per scheme: predicted size limit, observed size limit, critical path length
                        L* (where the goal's margin crosses zero, interpolated between sizes) and
                        the path length of the observed limit.
  3. decomposition.csv  per size: observed failures split into "incentive" (the goal is not the
                        best ending: no agent can learn it) and "learnability" (the goal is best,
                        the agent is too slow).
  4. genes.csv          among evolved schemes whose goal is the best ending: genes and margin of
                        those that learned vs those that failed (medians, A12, Mann-Whitney with
                        Holm), and a logistic regression over all sizes >= 2x.
  5. fig_size_curve.png/pdf     share of schemes that learn, and median episodes, vs maze size.
  6. fig_transfer_map.png/pdf   MAP-Elites map coloured by size limit: observed vs predicted
                                (median over runs), plus random search if present.

Examples (from the repository folder):
    python scaling/transfer.py --name transfer_all --dirs results
    python scaling/rule.py --name rule_all --dirs results --transfer results/scaling/transfer_all/transfer.csv
    python scaling/analyze_scaling.py --transfer results/scaling/transfer_all/transfer.csv \
                                      --rule results/scaling/rule_all/rule.csv

Only the normal drift (uniform 0.1, trained at the same drift) is used: the rule has no drift.
"""
import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import evolve  # noqa: E402

GENES = ["trap", "wall", "step", "progress", "delay"]
ALGO_SHORT = {"mapelites": "ME", "de": "DE", "random": "RS"}


# ---------------------------------------------------------------- helpers
def a12(x, y):
    """P(a value of x is larger than one of y), ties count 1/2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return ((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def num(v):
    return 999.0 if v == "end" else float(v)


def group_of(info):
    if info["kind"] == "baseline":
        return "hand-made"
    cond = info["condition"]
    prefix = "" if cond in ("single-maze", "") else f"{cond}, "
    return f"{prefix}{ALGO_SHORT.get(info['algo'], info['algo'])} {info['kind']}s"


def size_limit(ok):
    limit = 0
    for k in sorted(ok):
        if not ok[k]:
            break
        limit = k
    return limit


def logistic(X, y, l2=1e-3, iters=50):
    """Plain IRLS logistic regression with a small ridge penalty. Returns coefficients
    (intercept first). X is standardised by the caller, so the coefficients are comparable."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        H = Xb.T @ (Xb * W[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, Xb.T @ (y - p) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


# ---------------------------------------------------------------- loading
def load(transfer_csv, rule_csv, threshold):
    obs = defaultdict(lambda: defaultdict(list))         # key -> scale -> [(learned, n90)]
    info, path_of = {}, {}
    for r in csv.DictReader(open(transfer_csv)):
        if (r.get("drift") or "uniform") != "uniform" or float(r.get("slip") or 0.1) != 0.1 \
                or r.get("train_slip", "") not in ("", None):
            continue
        k = int(r["scale"])
        obs[r["candidate"]][k].append((r["learned"] == "True", float(r["n90"])))
        info[r["candidate"]] = {f: r[f] for f in ("source", "condition", "kind", "algo", "run", "cell",
                                                  "scheme", *GENES)}
    pred = defaultdict(dict)                              # key -> scale -> rule row
    for r in csv.DictReader(open(rule_csv)):
        pred[r["candidate"]][int(r["scale"])] = r
        path_of[int(r["scale"])] = int(r["shortest_path"])
    rows = []                                             # one per (scheme, size) present in both
    for key, by_scale in obs.items():
        if key not in pred:
            continue
        for k, v in by_scale.items():
            if k not in pred[key]:
                continue
            pr = pred[key][k]
            share = sum(l for l, _ in v) / len(v)
            rows.append(dict(key=key, scale=k, path=path_of[k], group=group_of(info[key]), **info[key],
                             n_seeds=len(v), learned_share=share, learned=share >= threshold,
                             n90=statistics.median(n for _, n in v), goal_best=pr["best_ending"] == "goal",
                             best_ending=pr["best_ending"], margin=float(pr["margin"])))
    missing = len([k for k in obs if k not in pred])
    if missing:
        print(f"!! {missing} schemes of transfer.csv are not in rule.csv (use the same --dirs/--runs/--elites)")
    if not rows:
        raise SystemExit("no scheme is in both files")
    return rows, path_of


# ---------------------------------------------------------------- 1. confusion
def confusion(rows, out):
    print("\n=== 1. rule prediction vs what the agents did ===")
    print("predicted: goal is the best ending (maze without slip);  observed: >= threshold of the seeds learned")
    print(f"{'size':>5s}{'schemes':>9s}{'pred ok & learned':>19s}{'pred ok & failed':>18s}"
          f"{'pred fail & failed':>20s}{'pred fail & learned':>21s}")
    table = []
    for k in sorted({r["scale"] for r in rows}) + ["all"]:
        rs = [r for r in rows if k == "all" or r["scale"] == k]
        c = dict(size=k, schemes=len(rs),
                 ok_learned=sum(r["goal_best"] and r["learned"] for r in rs),
                 ok_failed=sum(r["goal_best"] and not r["learned"] for r in rs),
                 fail_failed=sum(not r["goal_best"] and not r["learned"] for r in rs),
                 fail_learned=sum(not r["goal_best"] and r["learned"] for r in rs))
        n_fail_pred = c["fail_failed"] + c["fail_learned"]
        n_fail_obs = c["fail_failed"] + c["ok_failed"]
        c["fail_prediction_correct"] = round(c["fail_failed"] / n_fail_pred, 3) if n_fail_pred else ""
        c["failures_explained_by_rule"] = round(c["fail_failed"] / n_fail_obs, 3) if n_fail_obs else ""
        table.append(c)
        lab = f"{k}x" if k != "all" else "all"
        print(f"{lab:>5s}{c['schemes']:9d}{c['ok_learned']:19d}{c['ok_failed']:18d}{c['fail_failed']:20d}"
              f"{c['fail_learned']:21d}")
    t = table[-1]
    print(f"  'fails' predictions that came true: {t['fail_prediction_correct']}"
          f"   observed failures explained by the rule: {t['failures_explained_by_rule']}")
    print("  (a necessary condition: 'pred fail & learned' should be 0; 'pred ok & failed' = too slow, not the incentive)")
    write(out, "confusion.csv", table)


# ---------------------------------------------------------------- 2. limits and L*
def limits(rows, path_of, out):
    by = defaultdict(dict)
    for r in rows:
        by[r["key"]][r["scale"]] = r
    table = []
    for key, sc in by.items():
        ks = sorted(sc)
        pred_lim = size_limit({k: sc[k]["goal_best"] for k in ks})
        obs_lim = size_limit({k: sc[k]["learned"] for k in ks})
        # L*: first zero crossing of the margin, linear between the path lengths of two sizes
        lstar, censored = None, False
        for a, b in zip(ks, ks[1:]):
            ma, mb = sc[a]["margin"], sc[b]["margin"]
            if ma > 0 >= mb:
                la, lb = path_of[a], path_of[b]
                lstar = la + (lb - la) * ma / (ma - mb)
                break
        if lstar is None:
            if sc[ks[0]]["margin"] <= 0:
                lstar = path_of[ks[0]]
            else:
                lstar, censored = path_of[ks[-1]], True       # goal best at every tested size
        first = sc[ks[0]]
        table.append(dict(key=key, group=first["group"], scheme=first["scheme"], kind=first["kind"],
                          algo=first["algo"], run=first["run"], cell=first["cell"],
                          **{g: first[g] for g in GENES},
                          predicted_limit=pred_lim, observed_limit=obs_lim,
                          L_star=round(lstar, 1), L_star_censored=censored,
                          L_observed=path_of[obs_lim] if obs_lim else 0,
                          **{f"margin_{k}x": round(sc[k]["margin"], 4) for k in ks},
                          **{f"learned_{k}x": round(sc[k]["learned_share"], 2) for k in ks}))
    print("\n=== 2. size limits: predicted by the rule vs observed ===")
    ks = sorted(path_of)
    tested = sorted({r["scale"] for r in rows})
    print("rows = predicted limit, columns = observed limit (number of schemes)")
    print(f"{'pred/obs':>11s}" + "".join(f"{k:>6d}" for k in [0] + tested))
    for p in [0] + tested:
        print(f"{p:>11d}" + "".join(f"{sum(t['predicted_limit'] == p and t['observed_limit'] == o for t in table):6d}"
                                    for o in [0] + tested))
    above = sum(t["observed_limit"] > t["predicted_limit"] for t in table)
    same = sum(t["observed_limit"] == t["predicted_limit"] for t in table)
    print(f"  observed = predicted: {same}/{len(table)};  observed above predicted (rule wrong): {above};"
          f"  observed below (too slow): {len(table) - same - above}")
    fin = [t for t in table if not t["L_star_censored"]]
    if len(fin) >= 5:
        rho, p = spearmanr([t["L_star"] for t in fin], [t["L_observed"] for t in fin])
        print(f"  schemes with a finite L* (goal stops being best within {tested[-1]}x): {len(fin)};"
              f"  Spearman(L*, path length of the observed limit) = {rho:.2f} (p = {p:.2g})")
    else:
        print(f"  schemes with a finite L*: {len(fin)} (too few for a correlation)")
    write(out, "limits.csv", table)
    return table


# ---------------------------------------------------------------- 3 + 4. decomposition and genes
def decomposition(rows, out):
    print("\n=== 3. why do schemes fail? ===")
    print(f"{'size':>5s}{'group':>28s}{'schemes':>9s}{'failed':>8s}{'incentive':>11s}{'learnability':>14s}")
    table = []
    groups = sorted({r["group"] for r in rows}, key=lambda g: (g == "hand-made", g))
    for k in sorted({r["scale"] for r in rows}):
        for g in groups + ["all evolved"]:
            rs = [r for r in rows if r["scale"] == k and
                  (r["group"] == g or (g == "all evolved" and r["kind"] != "baseline"))]
            if not rs:
                continue
            fail = [r for r in rs if not r["learned"]]
            d = dict(size=k, group=g, schemes=len(rs), failed=len(fail),
                     incentive=sum(not r["goal_best"] for r in fail),
                     learnability=sum(r["goal_best"] for r in fail))
            table.append(d)
            print(f"{k:>4d}x{g[:28]:>28s}{d['schemes']:9d}{d['failed']:8d}{d['incentive']:11d}{d['learnability']:14d}")
    print("  incentive = the goal is not the best ending (no agent can learn it);"
          " learnability = the goal is best, the agent is too slow")
    write(out, "decomposition.csv", table)


def genes(rows, out, min_size=2):
    print(f"\n=== 4. evolved schemes whose goal is the best ending: learned vs failed (sizes >= {min_size}x) ===")
    ev = [r for r in rows if r["kind"] != "baseline" and r["goal_best"] and r["scale"] >= min_size]
    feats = GENES + ["margin"]
    tests = []
    for k in sorted({r["scale"] for r in ev}):
        rs = [r for r in ev if r["scale"] == k]
        L = [r for r in rs if r["learned"]]
        F = [r for r in rs if not r["learned"]]
        if len(L) < 3 or len(F) < 3:
            print(f"  {k}x: {len(L)} learned, {len(F)} failed: too few for a test")
            continue
        for f in feats:
            x, y = [num(r[f]) for r in L], [num(r[f]) for r in F]
            p = 1.0 if len(set(x + y)) == 1 else mannwhitneyu(x, y, alternative="two-sided").pvalue
            tests.append(dict(size=k, feature=f, n_learned=len(L), n_failed=len(F),
                              median_learned=statistics.median(x), median_failed=statistics.median(y),
                              A12_learned_larger=round(a12(x, y), 3), p=p))
    if tests:
        for t, pa in zip(tests, holm([t["p"] for t in tests])):
            t["p_holm"] = pa
            t["significant"] = pa < 0.05
        print(f"{'size':>5s}{'feature':>10s}{'median learned':>16s}{'median failed':>15s}{'A12':>7s}{'p_holm':>9s}")
        for t in tests:
            print(f"{t['size']:>4d}x{t['feature']:>10s}{t['median_learned']:16.3f}{t['median_failed']:15.3f}"
                  f"{t['A12_learned_larger']:7.2f}{t['p_holm']:9.3g}{' *' if t['significant'] else ''}")
        print("  A12 = P(the feature is larger in a scheme that learned than in one that failed);"
              " delay 'end' = 999; * = p_holm < 0.05")
        write(out, "genes.csv", tests)
    # logistic regression over all sizes >= min_size: which features predict learning?
    if len(ev) >= 20 and 0 < sum(r["learned"] for r in ev) < len(ev):
        names = feats + ["path"]
        X = np.array([[num(r[f]) for f in feats] + [r["path"]] for r in ev])
        sd = X.std(axis=0)
        keep = sd > 0
        Xs = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]
        y = np.array([r["learned"] for r in ev], float)
        w = logistic(Xs, y)
        print(f"  logistic regression (learned ~ standardised features), {len(ev)} cases:")
        coef = [(n, c) for n, c in zip([n for n, k in zip(names, keep) if k], w[1:])]
        print("   " + "  ".join(f"{n} {c:+.2f}" for n, c in coef))
        print("   positive = a larger value makes learning more likely (comparable sizes: features are standardised)")
        write(out, "logistic.csv", [dict(feature=n, coefficient=round(c, 4)) for n, c in coef])


# ---------------------------------------------------------------- figures
def figures(rows, lims, path_of, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted({r["scale"] for r in rows})
    colours = {"ME": "#2a6fdb", "DE": "#e8902a", "RS": "#2a9d8f"}
    groups = sorted({r["group"] for r in rows}, key=lambda g: (g == "hand-made", g))
    # size curve
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        algo = g.split(",")[-1].strip().split(" ")[0]
        col = colours.get(algo, "#333333")
        ls = "-" if "winner" in g else ":" if "elite" in g else "--"
        share = [np.mean([r["learned"] for r in rs if r["scale"] == k]) for k in ks]
        a1.plot([path_of[k] for k in ks], share, ls, marker="o", ms=3, color=col,
                label=f"{g} ({len({r['key'] for r in rs})})")
        if "elite" not in g and g != "hand-made":
            med = [statistics.median(r["n90"] for r in rs if r["scale"] == k) for k in ks]
            a2.plot([path_of[k] for k in ks], med, ls, marker="o", ms=3, color=col, label=g)
    for b in sorted({r["scheme"] for r in rows if r["kind"] == "baseline"}):
        rs = [r for r in rows if r["scheme"] == b and r["kind"] == "baseline"]
        if any(r["learned"] for r in rs):
            med = [statistics.median(r["n90"] for r in rs if r["scale"] == k) for k in ks]
            a2.plot([path_of[k] for k in ks], med, "--", marker=".", lw=1, color="#bbbbbb")
            a2.annotate(b.split("_")[0], (path_of[ks[-1]], med[-1]), fontsize=6, color="#888888")
    a1.set_xlabel("shortest path (steps)")
    a1.set_ylabel("share of schemes that learn")
    a1.set_ylim(-0.02, 1.02)
    a1.legend(fontsize=6)
    a2.set_xlabel("shortest path (steps)")
    a2.set_ylabel("median episodes to 90%")
    a2.set_title("winners (coloured) and hand-made (grey)", fontsize=8)
    for a in (a1, a2):
        a.set_xticks([path_of[k] for k in ks], [f"{path_of[k]}\n({k}x)" for k in ks], fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_size_curve.{ext}"), dpi=150)
    plt.close(fig)
    # transfer map
    panels = []
    for algo in ("mapelites", "random"):
        el = [t for t in lims if t["kind"] == "elite" and t["algo"] == algo and t["cell"]]
        if el:
            panels.append((f"{ALGO_SHORT[algo]} elites: observed", el, "observed_limit"))
            if algo == "mapelites":
                panels.append(("ME elites: predicted by the rule", el, "predicted_limit"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.1), squeeze=False)
    vmax = max(ks)
    for ax, (title, el, field) in zip(axes[0], panels):
        cells = defaultdict(list)
        for t in el:
            i, j = map(int, t["cell"].split("-"))
            cells[(i, j)].append(t[field])
        grid = np.full(evolve.GRID, np.nan)
        for (i, j), v in cells.items():
            grid[i, j] = statistics.median(v)
        im = ax.imshow(grid, origin="lower", extent=[0, evolve.MAX_DELAY, 0, 1], aspect="auto",
                       cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{title}\n(median over {len({t['run'] for t in el})} runs)", fontsize=8)
        ax.set_xlabel("mean feedback delay (steps)", fontsize=8)
        ax.set_ylabel("punishment share", fontsize=8)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85)
    cb.set_label("size limit (0 = not even 1x)", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_transfer_map.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write(out, name, data):
    if not data:
        return
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for d in data for k in d)))
        w.writeheader()
        w.writerows(data)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transfer", required=True, help="transfer.csv from scaling/transfer.py")
    p.add_argument("--rule", required=True, help="rule.csv from scaling/rule.py (same --dirs/--runs/--elites)")
    p.add_argument("--threshold", type=float, default=0.8, help="'learns a size', as in transfer.py")
    p.add_argument("--out", help="default: <folder of transfer.csv>/analysis")
    a = p.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.transfer)), "analysis")
    os.makedirs(out, exist_ok=True)
    rows, path_of = load(a.transfer, a.rule, a.threshold)
    print(f"{len({r['key'] for r in rows})} schemes, sizes {sorted({r['scale'] for r in rows})}, "
          f"{len(rows)} (scheme, size) pairs")
    confusion(rows, out)
    lims = limits(rows, path_of, out)
    decomposition(rows, out)
    genes(rows, out)
    figures(rows, lims, path_of, out)
    print(f"\nsaved tables and figures in {out}")


if __name__ == "__main__":
    main()
