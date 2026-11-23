"""Step 3: how far do reward schemes from the small maze scale?

Take saved reward schemes and train NEW agents with them on bigger copies of the training maze
(dyna_9x6_3traps scaled 1x to 5x: same layout, every cell a k x k block). The reward scheme is
fixed; only the agent learns. For every scheme this gives its "size limit": the biggest copy it
still learns.

Schemes (candidates):
    winner     the best scheme of every run in the given results folders (MAP-Elites, DE,
               random search; single-maze, multi-maze or variant runs)
    elite      every scheme in the MAP-Elites / random-search maps of those runs (by default only
               the elites that learned on every evaluation seed of their training set)
    baseline   the hand-made and published schemes (rewards.baseline_schemes, built for each
               maze, so the "matched" ones are matched to that maze's path length)

Run from the repository folder:
    python scaling/transfer.py                                        # saved single-maze runs, 1x-4x
    python scaling/transfer.py --runs 7 --seeds 3 --scales 1 2 3      # quick pilot
    python scaling/transfer.py --dirs results/2026-11-15_laptop results/multimaze/A --name single_vs_multi
    python scaling/transfer.py --summarize results/scaling/transfer/transfer.csv    # tables only

Output: results/scaling/<name>/transfer.csv (one row per scheme, maze and seed; written while
the runs finish, so --resume continues an interrupted job) and summary.csv (one row per scheme:
genes, learning per size, size limit).

Choices (all options):
    --scales 1 2 3 4         sizes to test (5x is also available)
    --episodes-per-scale 500 training episodes = 500 x scale (500, 1000, 1500, 2000, 2500), so a
                             bigger maze gets proportionally more exploration; --fixed-episodes N
                             uses N for every size instead
    --seeds 10               new seeds per scheme and size (6000 and up; never used before), the
                             same seeds for every scheme
    --threshold 0.8          a scheme "learns" a size if at least 80% of its seeds reach 90% success
    --elites learning        learning (default), all, or none
"""
import argparse
import csv
import functools
import glob
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evolve  # noqa: E402
from gridmaze import get_maze  # noqa: E402
from qlearning import train  # noqa: E402
from rewards import baseline_schemes  # noqa: E402

SCALED = {1: "dyna_9x6_3traps", 2: "dyna_18x12_3traps", 3: "dyna_27x18_3traps",
          4: "dyna_36x24_3traps", 5: "dyna_45x30_3traps"}
SEEDS_START = 6000                    # evolution 0-11, re-evaluation 1000+, figures 2000-5999
MAIN_BASELINES = ["A1", "A2", "B1_immediate", "B2", "sparse_dyna", "russell_norvig_matched"]
GENES = ["trap", "wall", "step", "progress", "delay"]
ALGO_SHORT = {"mapelites": "ME", "de": "DE", "random": "RS"}


# ---------------------------------------------------------------- candidates
def _decode(genome, bounds, name):
    """Decode with the gene ranges the run was evolved with (variant runs store their own)."""
    if not bounds:
        return evolve.decode(genome, name=name)
    saved = dict(evolve.BOUNDS)
    evolve.BOUNDS.update({g: tuple(v) for g, v in bounds.items()})
    try:
        return evolve.decode(genome, name=name)
    finally:
        evolve.BOUNDS.clear()
        evolve.BOUNDS.update(saved)


def load_candidates(dirs, elites="learning", runs=None):
    """Winners and elites of every run file in the given results folders."""
    cands = []
    for d in dirs:
        evo = os.path.join(d, "evolution")
        evo = evo if os.path.isdir(evo) else d
        source = os.path.basename(os.path.normpath(d))
        files = sorted(glob.glob(os.path.join(evo, "*_run*.json")))
        if not files:
            print(f"!! no run files in {evo}")
        for path in files:
            with open(path) as f:
                r = json.load(f)
            if runs is not None and r["run"] not in runs:
                continue
            condition = r.get("condition") or (f"variant {r['variant']}" if r.get("variant") else "single-maze")
            algo, bounds = r["algo"], r.get("bounds")
            base = dict(source=source, condition=condition, algo=algo, run=r["run"])
            tag = f"{ALGO_SHORT.get(algo, algo)}{r['run']}"
            cands.append(dict(base, kind="winner", cell="", train_fitness=r["best"]["fitness"],
                              scheme=_decode(r["best"]["genome"], bounds, f"{tag}_winner")))
            if elites == "none" or "archive" not in r:
                continue
            for e in r["archive"]:
                if e["genome"] == r["best"]["genome"]:
                    continue                          # already in as the winner
                if elites == "learning" and e["converged"] < len(e["n90"]):
                    continue
                cell = "-".join(map(str, e["cell"]))
                cands.append(dict(base, kind="elite", cell=cell, train_fitness=e["fitness"],
                                  scheme=_decode(e["genome"], bounds, f"{tag}_cell{cell}")))
    return cands


def candidate_key(c):
    return f"{c['source']}|{c['kind']}|{c['scheme'].name}"


# ---------------------------------------------------------------- training
@functools.lru_cache(maxsize=None)
def _maze(name):
    return get_maze(name)


def _job(job):
    key, scale, maze_name, episodes, seed, scheme = job
    maze = _maze(maze_name)
    r = train(maze, scheme, seed, episodes=episodes)
    n = r["episodes_to_90"]
    return dict(candidate=key, scale=scale, maze=maze_name, episodes=episodes, seed=seed,
                slip=maze.slip, learned=n is not None, n90=n if n is not None else 2 * episodes,
                final_success=round(r["final_success"], 3), trap_falls=r["trap_falls"])


def cand_fields(c):
    s = c["scheme"]
    return dict(source=c["source"], condition=c["condition"], kind=c["kind"], algo=c["algo"],
                run=c["run"], cell=c["cell"], scheme=s.name, train_fitness=c["train_fitness"],
                **{g: (round(getattr(s, g), 4) if g != "delay" else s.delay) for g in GENES})


# ---------------------------------------------------------------- summary
def summarize(csv_path, threshold, out_dir=None):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise SystemExit(f"{csv_path} is empty")
    by = defaultdict(lambda: defaultdict(list))
    info = {}
    for r in rows:
        by[r["candidate"]][int(r["scale"])].append((r["learned"] == "True", float(r["n90"])))
        info[r["candidate"]] = {k: r[k] for k in ("source", "condition", "kind", "algo", "run", "cell",
                                                  "scheme", "train_fitness", *GENES)}
    scales = sorted({int(r["scale"]) for r in rows})
    per = []
    for key, sc in by.items():
        row = dict(info[key])
        limit, ok_so_far = 0, True
        for k in scales:
            v = sc.get(k, [])
            share = sum(l for l, _ in v) / len(v) if v else float("nan")
            row[f"learned_{k}x"] = round(share, 2) if v else ""
            row[f"n90_{k}x"] = statistics.median(n for _, n in v) if v else ""
            if v and ok_so_far and share >= threshold:
                limit = k
            elif v:
                ok_so_far = False
        row["size_limit"] = limit
        per.append(row)
    if out_dir:
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per[0]))
            w.writeheader()
            w.writerows(per)

    sources_of = defaultdict(set)              # name a group by its condition, plus the folder if needed
    for r in per:
        sources_of[r["condition"]].add(r["source"])

    def group(r):
        if r["kind"] == "baseline":
            return ("hand-made", r["scheme"])
        label = r["condition"] if len(sources_of[r["condition"]]) == 1 else f"{r['condition']} ({r['source']})"
        return (label, f"{ALGO_SHORT.get(r['algo'], r['algo'])} {r['kind']}s")
    groups = defaultdict(list)
    for r in per:
        groups[group(r)].append(r)

    print(f"\nA scheme learns a size if at least {threshold:.0%} of its seeds reach 90% success.")
    print("Size limit = the biggest size it learns, with every smaller size learned too (0 = not even 1x).\n")
    head = "".join(f"{f'{k}x':>12s}" for k in scales)
    print(f"{'group':52s}{'schemes':>8s}{head}{'size limit (median)':>22s}")
    print(f"{'':52s}{'':>8s}" + "".join(f"{'learn / N90':>12s}" for _ in scales))
    order = sorted(groups, key=lambda g: (g[0] == "hand-made", g))
    for g in order:
        rs = groups[g]
        cells = []
        for k in scales:
            sh = [r[f"learned_{k}x"] for r in rs if r[f"learned_{k}x"] != ""]
            nn = [r[f"n90_{k}x"] for r in rs if r[f"n90_{k}x"] != ""]
            if not sh:
                cells.append(f"{'-':>12s}")
                continue
            frac = sum(s >= threshold for s in sh) / len(sh)
            cells.append(f"{frac:5.0%} /{statistics.median(nn):5.0f}")
        name = f"{g[0]}, {g[1]}" if g[0] != "hand-made" else f"hand-made: {g[1]}"
        lim = statistics.median(r["size_limit"] for r in rs)
        print(f"{name[:52]:52s}{len(rs):8d}" + "".join(f"{c:>12s}" for c in cells) + f"{lim:22.1f}")
    print("\nlearn = share of the schemes in the group that learn that size;"
          " N90 = median over the group of each scheme's median episodes to 90% (2 x episodes = never)")
    if out_dir:
        print(f"saved {os.path.join(out_dir, 'summary.csv')}")
    return per


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", default=[os.path.join(ROOT, "results", "2026-11-15_laptop")],
                   help="results folders with evolution/*.json (default: the saved single-maze runs)")
    p.add_argument("--runs", nargs="+", type=int, help="only these run numbers (default: all)")
    p.add_argument("--elites", choices=["learning", "all", "none"], default="learning")
    p.add_argument("--baselines", choices=["main", "all", "none"], default="main")
    p.add_argument("--scales", nargs="+", type=int, default=[1, 2, 3, 4], choices=sorted(SCALED))
    p.add_argument("--episodes-per-scale", type=int, default=500)
    p.add_argument("--fixed-episodes", type=int, help="use this many episodes for every size instead")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--name", default="transfer", help="output folder results/scaling/<name>")
    p.add_argument("--resume", action="store_true", help="skip the runs already in transfer.csv")
    p.add_argument("--summarize", metavar="CSV", help="only print the tables of an existing transfer.csv")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    if a.summarize:
        summarize(a.summarize, a.threshold, os.path.dirname(a.summarize))
        return

    out_dir = os.path.join(ROOT, "results", "scaling", a.name)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "transfer.csv")

    cands = load_candidates(a.dirs, a.elites, a.runs)
    seeds = list(range(SEEDS_START, SEEDS_START + a.seeds))
    jobs, fields = [], {}
    for k in a.scales:
        maze_name = SCALED[k]
        episodes = a.fixed_episodes or a.episodes_per_scale * k
        here = list(cands)
        if a.baselines != "none":
            for s in baseline_schemes(_maze(maze_name)):
                if a.baselines == "all" or any(s.name.startswith(b) for b in MAIN_BASELINES):
                    here.append(dict(source="hand-made", condition="hand-made", algo="", run="",
                                     kind="baseline", cell="", train_fitness="", scheme=s))
        for c in here:
            key = candidate_key(c)
            fields[key] = cand_fields(c)
            jobs += [(key, k, maze_name, episodes, seed, c["scheme"]) for seed in seeds]

    done = set()
    if a.resume and os.path.exists(csv_path):
        with open(csv_path) as f:
            done = {(r["candidate"], int(r["scale"]), int(r["seed"])) for r in csv.DictReader(f)}
        jobs = [j for j in jobs if (j[0], j[1], j[4]) not in done]
    per_scale = defaultdict(int)
    for j in jobs:
        per_scale[j[1]] += 1
    n_cands = len({j[0] for j in jobs})
    print(f"{len(cands)} evolved schemes from {len(a.dirs)} folder(s) + baselines; {n_cands} schemes to run")
    print("training runs per size: " + ", ".join(f"{k}x: {per_scale[k]}" for k in sorted(per_scale))
          + (f"  ({len(done)} already done)" if done else ""))
    if not jobs:
        summarize(csv_path, a.threshold, out_dir)
        return

    columns = ["candidate", "source", "condition", "kind", "algo", "run", "cell", "scheme", "train_fitness",
               *GENES, "scale", "maze", "episodes", "seed", "slip", "learned", "n90", "final_success", "trap_falls"]
    new_file = not (a.resume and os.path.exists(csv_path))
    t0, n = time.time(), 0
    with open(csv_path, "w" if new_file else "a", newline="") as f, mp.Pool(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            w.writeheader()
        for res in pool.imap_unordered(_job, jobs, chunksize=1):
            w.writerow({**fields[res["candidate"]], **res})
            n += 1
            if n % 200 == 0 or n == len(jobs):
                f.flush()
                el = time.time() - t0
                print(f"[{el:6.0f}s] {n}/{len(jobs)} training runs  (rough estimate: {el / n * (len(jobs) - n) / 60:.0f} min left; bigger sizes come last)",
                      flush=True)
    print(f"saved {csv_path}")
    summarize(csv_path, a.threshold, out_dir)


if __name__ == "__main__":
    main()
