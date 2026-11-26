"""Scaling thesis, part 3: is the MAP-Elites map an insurance when the maze grows?

The winner of a run is the fastest scheme on the SMALL maze. On a bigger maze another scheme of
the same map may be better. This script tests that without cheating:

  1. selection: on the selection size (default 2x), every scheme of a run's map (MAP-Elites
     elites, random-search map, DE final population) is trained on the selection seeds, and the
     best one is the run's "pick" (most seeds learned, then fewest episodes). By default the
     selection reuses transfer.py's results (seeds 6000+), so nothing is trained twice.
  2. test: the picks, the small-maze winners and the hand-made baselines are trained on the
     bigger test sizes (default 3x and 4x) with FRESH seeds (8000+), never used for choosing.

Groups compared on the test sizes:
    ME pick, RS pick, DE pick     chosen on 2x from the map / final population of each run
    ME winner, DE winner, RS winner   the small-maze winner of each run (what you get without a map)
    hand-made baselines           (default B1, A1, Russell & Norvig matched)

Statistics, per test size (unit = one run: the mean episodes of its scheme over the test seeds):
    ME pick vs ME winner          paired by run (Wilcoxon signed-rank): does the map beat its own winner?
    ME pick vs DE winner / DE pick, ME pick vs RS pick   Mann-Whitney U
    every pick vs each baseline   per scheme, test seeds vs test seeds (Mann-Whitney U),
                                  counted as faster / no difference / slower
All p-values Holm-corrected together; A12 > 0.5 = the first group needs fewer episodes.

Run from the repository folder:
    python scaling/archive_pick.py --dirs results --transfer results/scaling/transfer_all/transfer.csv
    python scaling/archive_pick.py --runs 7 --seeds 3 --transfer results/scaling/pilot/transfer.csv --name pick_pilot

Output in results/scaling/<name>/: picks.csv (which scheme each run picked, genes, selection
score), test.csv (one row per scheme, size and seed), stats.csv, and the printed tables.
"""
import argparse
import csv
import json
import glob
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import transfer  # noqa: E402
from rewards import baseline_schemes  # noqa: E402

TEST_SEEDS_START = 8000            # never used elsewhere (transfer 6000+, references 7000+)
GENES = transfer.GENES


def a12_smaller(x, y):
    """P(a value of x is SMALLER than one of y): > 0.5 = x needs fewer episodes."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return ((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mwu(x, y):
    if len(set(x) | set(y)) == 1:
        return 1.0
    return mannwhitneyu(x, y, alternative="two-sided").pvalue


# ---------------------------------------------------------------- the maps of every run
def run_maps(dirs, runs=None, elites="learning"):
    """run id -> dict(algo, run, source, winner candidate, map candidates). DE's final population
    plays the role of its map (DE keeps no archive)."""
    maps = {}
    cands = transfer.load_candidates(dirs, elites, runs)
    for c in cands:
        rid = (c["source"], c["condition"], c["algo"], c["run"])
        m = maps.setdefault(rid, dict(source=c["source"], condition=c["condition"], algo=c["algo"],
                                      run=c["run"], winner=None, map=[]))
        if c["kind"] == "winner":
            m["winner"] = c
        m["map"].append(c)                                # the winner belongs to its own map too
    # DE: add the final population as its "map"
    for d in dirs:
        evo = os.path.join(d, "evolution")
        evo = evo if os.path.isdir(evo) else d
        source = os.path.basename(os.path.normpath(d))
        for path in sorted(glob.glob(os.path.join(evo, "de_run*.json"))):
            r = json.load(open(path))
            if runs is not None and r["run"] not in runs or "population" not in r:
                continue
            condition = r.get("condition") or (f"variant {r['variant']}" if r.get("variant") else "single-maze")
            rid = (source, condition, "de", r["run"])
            if rid not in maps:
                continue
            for i, e in enumerate(r["population"]):
                if elites == "learning" and e["converged"] < len(e["n90"]):
                    continue
                if e["genome"] == r["best"]["genome"]:
                    continue
                sch = transfer._decode(e["genome"], r.get("bounds"), f"DE{r['run']}_pop{i}")
                maps[rid]["map"].append(dict(source=source, condition=condition, algo="de", run=r["run"],
                                             kind="population", cell="", train_fitness=e["fitness"], scheme=sch))
    return maps


# ---------------------------------------------------------------- training jobs
def _job(job):
    tag, key, scale, seed, scheme = job
    maze_name = transfer.SCALED[scale]
    episodes = 500 * scale
    r = transfer.train(transfer._maze(maze_name), scheme, seed, episodes=episodes)
    n = r["episodes_to_90"]
    return dict(tag=tag, candidate=key, scale=scale, maze=maze_name, episodes=episodes, seed=seed,
                learned=n is not None, n90=n if n is not None else 2 * episodes)


def run_jobs(jobs, workers, label):
    out, t0 = [], time.time()
    if not jobs:
        return out
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, jobs, chunksize=1), 1):
            out.append(res)
            if i % 200 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  [{label} {el:6.0f}s] {i}/{len(jobs)}  (about {el / i * (len(jobs) - i) / 60:.0f} min left)",
                      flush=True)
    return out


def score(results):
    """(share learned, mean episodes) -> sort key: more learned first, then fewer episodes."""
    share = np.mean([r["learned"] for r in results])
    return (-share, float(np.mean([r["n90"] for r in results])))


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", default=[os.path.join(ROOT, "results", "2026-11-15_laptop")])
    p.add_argument("--runs", nargs="+", type=int, help="only these run numbers (default: all)")
    p.add_argument("--select-scale", type=int, default=2)
    p.add_argument("--test-scales", nargs="+", type=int, default=[3, 4])
    p.add_argument("--transfer", help="transfer.csv to reuse for the selection (its seeds at --select-scale)")
    p.add_argument("--select-seeds", type=int, default=10, help="only without --transfer (seeds 6000+)")
    p.add_argument("--seeds", type=int, default=20, help="fresh test seeds per scheme and size (8000+)")
    p.add_argument("--baselines", nargs="*", default=["B1_immediate", "A1", "russell_norvig_matched"])
    p.add_argument("--name", default="archive_pick")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    if a.select_scale in a.test_scales:
        raise SystemExit("the selection size must not be a test size")
    out = os.path.join(ROOT, "results", "scaling", a.name)
    os.makedirs(out, exist_ok=True)

    maps = run_maps(a.dirs, a.runs)
    print(f"{len(maps)} runs: " + ", ".join(f"{transfer.ALGO_SHORT.get(m['algo'], m['algo'])}{m['run']} "
                                            f"({len(m['map'])} schemes)" for m in maps.values()))

    # 1. selection -------------------------------------------------------------
    sel = defaultdict(list)                                  # candidate key -> results at select size
    if a.transfer:
        for r in csv.DictReader(open(a.transfer)):
            if int(r["scale"]) == a.select_scale and (r.get("drift") or "uniform") == "uniform" \
                    and float(r.get("slip") or 0.1) == 0.1 and r.get("train_slip", "") in ("", None):
                sel[r["candidate"]].append(dict(learned=r["learned"] == "True", n90=float(r["n90"])))
    need = [c for m in maps.values() for c in m["map"] if transfer.candidate_key(c) not in sel]
    if need:
        print(f"selection: {len(need)} schemes not in the transfer file, training them on {a.select_scale}x"
              f" with {a.select_seeds} seeds (6000+)")
        jobs = [("select", transfer.candidate_key(c), a.select_scale, s, c["scheme"])
                for c in need for s in range(transfer.SEEDS_START, transfer.SEEDS_START + a.select_seeds)]
        for r in run_jobs(jobs, a.workers, "selection"):
            sel[r["candidate"]].append(r)
    picks = []
    for m in maps.values():
        best = min(m["map"], key=lambda c: score(sel[transfer.candidate_key(c)]))
        sh, mean_n = score(sel[transfer.candidate_key(best)])
        w_sh, w_n = score(sel[transfer.candidate_key(m["winner"])])
        short = transfer.ALGO_SHORT.get(m["algo"], m["algo"])
        picks.append(dict(source=m["source"], condition=m["condition"], algo=m["algo"], run=m["run"],
                          group_pick=f"{short} pick", group_winner=f"{short} winner",
                          pick=best["scheme"].name, pick_kind=best["kind"], pick_cell=best["cell"],
                          pick_is_winner=best is m["winner"], map_size=len(m["map"]),
                          select_learned=-sh, select_mean_n90=round(mean_n, 1),
                          winner_select_learned=-w_sh, winner_select_mean_n90=round(w_n, 1),
                          **{g: getattr(best["scheme"], g) for g in GENES},
                          _pick=best, _winner=m["winner"]))
    print(f"\nselection on {a.select_scale}x (share of seeds learned / mean episodes):")
    for pk in sorted(picks, key=lambda x: (x["algo"], x["run"])):
        same = " (= winner)" if pk["pick_is_winner"] else ""
        print(f"  {pk['group_pick']:8s} run {pk['run']}: pick {pk['pick']:22s} {pk['select_learned']:.0%} /"
              f" {pk['select_mean_n90']:6.0f}{same}   winner {pk['winner_select_learned']:.0%} / {pk['winner_select_mean_n90']:6.0f}")
    with open(os.path.join(out, "picks.csv"), "w", newline="") as f:
        cols = [k for k in picks[0] if not k.startswith("_")]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(picks)

    # 2. test on bigger sizes with fresh seeds ------------------------------------
    seeds = list(range(TEST_SEEDS_START, TEST_SEEDS_START + a.seeds))
    schemes = {}                                            # (group, run) -> scheme
    for pk in picks:
        schemes[(pk["group_pick"], pk["run"])] = pk["_pick"]["scheme"]
        schemes[(pk["group_winner"], pk["run"])] = pk["_winner"]["scheme"]
    jobs = []
    for (g, run), sch in schemes.items():
        for k in a.test_scales:
            jobs += [(g, f"{g}|{run}", k, s, sch) for s in seeds]
    for k in a.test_scales:
        for b in baseline_schemes(transfer._maze(transfer.SCALED[k])):
            if any(b.name.startswith(x) for x in a.baselines):
                jobs += [(b.name, f"{b.name}|", k, s, b) for s in seeds]
    print(f"\ntest: {len(jobs)} training runs on {', '.join(f'{k}x' for k in a.test_scales)}, seeds {seeds[0]}-{seeds[-1]}")
    res = run_jobs(jobs, a.workers, "test")
    with open(os.path.join(out, "test.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(sorted(res, key=lambda r: (r["candidate"], r["scale"], r["seed"])))

    # 3. tables and tests --------------------------------------------------------
    per = defaultdict(lambda: defaultdict(list))            # scale -> candidate -> n90 list
    for r in res:
        per[r["scale"]][r["candidate"]].append(r["n90"])
    learned = defaultdict(lambda: defaultdict(list))
    for r in res:
        learned[r["scale"]][r["tag"]].append(r["learned"])
    stats = []
    for k in a.test_scales:
        print(f"\n=== {k}x ({transfer.SCALED[k]}), {a.seeds} fresh seeds; 'never' = {2 * 500 * k} episodes ===")
        runs_of = defaultdict(dict)                          # group -> run -> mean n90
        for cand, v in per[k].items():
            g, run = cand.split("|")
            runs_of[g][run] = float(np.mean(v))
        order = sorted(runs_of, key=lambda g: statistics.median(runs_of[g].values()))
        print(f"  {'group':28s}{'runs':>6s}{'median of run means':>21s}{'seeds learned':>15s}")
        for g in order:
            print(f"  {g:28s}{len(runs_of[g]):6d}{statistics.median(runs_of[g].values()):21.0f}"
                  f"{np.mean(learned[k][g]):15.0%}")
        me_pick, me_win = runs_of.get("ME pick", {}), runs_of.get("ME winner", {})
        common = sorted(set(me_pick) & set(me_win))
        if len(common) >= 2:
            x, y = [me_pick[r] for r in common], [me_win[r] for r in common]
            d = np.array(x) - np.array(y)
            pv = 1.0 if np.all(d == 0) else wilcoxon(x, y).pvalue
            stats.append(dict(size=k, A="ME pick", B="ME winner", unit="run (paired)", n_A=len(x), n_B=len(y),
                              median_A=statistics.median(x), median_B=statistics.median(y),
                              A12=a12_smaller(x, y), p=pv))
        for A, B in (("ME pick", "DE winner"), ("ME pick", "DE pick"), ("ME pick", "RS pick"),
                     ("RS pick", "RS winner"), ("DE pick", "DE winner")):
            if A in runs_of and B in runs_of and len(runs_of[A]) >= 2 and len(runs_of[B]) >= 2:
                x, y = list(runs_of[A].values()), list(runs_of[B].values())
                stats.append(dict(size=k, A=A, B=B, unit="run", n_A=len(x), n_B=len(y),
                                  median_A=statistics.median(x), median_B=statistics.median(y),
                                  A12=a12_smaller(x, y), p=mwu(x, y)))
        for cand, v in per[k].items():
            g, run = cand.split("|")
            if not g.endswith("pick"):
                continue
            for b in (c for c in per[k] if c.endswith("|")):
                bv = per[k][b]
                stats.append(dict(size=k, A=cand, B=b.rstrip("|"), unit="seed", n_A=len(v), n_B=len(bv),
                                  median_A=statistics.median(v), median_B=statistics.median(bv),
                                  A12=a12_smaller(v, bv), p=mwu(v, bv)))
    if stats:
        for s, pa in zip(stats, holm([s["p"] for s in stats])):
            s["p_holm"], s["significant"] = pa, pa < 0.05
            s["verdict"] = ("faster" if s["A12"] > 0.5 else "slower") if s["significant"] else "no difference"
        with open(os.path.join(out, "stats.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stats[0]))
            w.writeheader()
            w.writerows(stats)
        print(f"\n=== tests, Holm-corrected over {len(stats)} tests (A12 > 0.5: A needs fewer episodes) ===")
        for s in stats:
            if s["unit"] != "seed":
                print(f"  {s['size']}x  {s['A']:8s} vs {s['B']:10s} ({s['unit']:12s}) median {s['median_A']:6.0f} vs"
                      f" {s['median_B']:6.0f}  A12 {s['A12']:.2f}  p_holm {s['p_holm']:.3g}{' *' if s['significant'] else ''}")
        print("  picks vs each baseline (one test per pick, seeds vs seeds): faster / no difference / slower")
        for k in a.test_scales:
            for g in ("ME pick", "RS pick", "DE pick"):
                for b in sorted({s["B"] for s in stats if s["unit"] == "seed"}):
                    ss = [s for s in stats if s["size"] == k and s["unit"] == "seed" and s["B"] == b
                          and s["A"].startswith(g + "|")]
                    if ss:
                        c = {v: sum(s["verdict"] == v for s in ss) for v in ("faster", "no difference", "slower")}
                        print(f"  {k}x  {g:8s} vs {b:28s} {c['faster']:3d} / {c['no difference']:3d} / {c['slower']:3d}")
    print(f"\nsaved picks.csv, test.csv, stats.csv in {out}")


if __name__ == "__main__":
    main()
