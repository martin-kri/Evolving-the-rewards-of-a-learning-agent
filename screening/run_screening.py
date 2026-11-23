"""Screening: try many configurations cheaply, then pick the ones worth full runs.

For every configuration (screening/configs.py) and algorithm, a few short runs (default: 3 runs,
budget 2000) are made; the winner of every run is re-evaluated on the VALIDATION mazes (new
seeds); then the configurations are compared. The unseen_* test mazes and the scaled copies are
not used here, so the final comparison stays clean.

Run from the repository folder:
    python screening/run_screening.py all                       # run + reeval + compare, default configs
    python screening/run_screening.py run --configs single A A_worst --algos mapelites
    python screening/run_screening.py reeval
    python screening/run_screening.py compare
    python screening/run_screening.py run --print-commands      # only print the commands (server, parallel)
    python screening/run_screening.py compare --score relative  # every validation maze weighs the same
    python screening/run_screening.py list                      # the configurations and the time they take

Every step skips what already exists, so it can be stopped and started again. Results go to
results/screening/<configuration>/ (evolution/, reevaluation.csv), the comparison to
results/screening/comparison_<score>.csv.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import configs  # noqa: E402

OUT = os.path.join(ROOT, "results", "screening")
ALGO_SHORT = {"mapelites": "ME", "de": "DE", "random": "RS"}
LABEL_ALGO = {"ME best": "mapelites", "DE best": "de", "RS best": "random"}
# seconds per single-maze evaluation on the laptop (saved runs: MAP-Elites ~1040 s, DE ~315 s for
# 5000 evaluations; random search assumed a bit slower than MAP-Elites, its schemes fail more often)
SEC_PER_EVAL = {"mapelites": 0.21, "de": 0.065, "random": 0.25}


def folder(name):
    return os.path.join(OUT, name)


def run_commands(a):
    cmds = []
    for name in a.configs:
        c = configs.CONFIGS[name]
        for algo in a.algos:
            for r in range(a.runs):
                out = os.path.join(folder(name), "evolution")
                if os.path.exists(os.path.join(out, f"{algo}_run{r}.json")):
                    continue
                cmd = [sys.executable, os.path.join(ROOT, "multimaze", "run_multimaze.py"),
                       "--set", c["set"], "--combine", c["combine"], "--variant", c["variant"],
                       "--algo", algo, "--run", str(r), "--budget", str(a.budget), "--out", out]
                if a.workers:
                    cmd += ["--workers", str(a.workers)]
                if a.quick:
                    cmd += ["--quick"]
                cmds.append((name, algo, r, cmd))
    return cmds


def step_run(a):
    cmds = run_commands(a)
    if a.print_commands:
        for *_, cmd in cmds:
            print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
        return
    print(f"{len(cmds)} runs to do")
    for i, (name, algo, r, cmd) in enumerate(cmds, 1):
        print(f"\n=== [{i}/{len(cmds)}] {name}: {algo} run {r} ===", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)


def step_reeval(a):
    for name in a.configs:
        d = folder(name)
        csv_path = os.path.join(d, "reevaluation.csv")
        if not glob.glob(os.path.join(d, "evolution", "*_run*.json")):
            print(f"{name}: no runs yet, skipped")
            continue
        if os.path.exists(csv_path) and not a.force:
            print(f"{name}: reevaluation.csv exists (use --force to redo)")
            continue
        print(f"\n=== {name}: re-evaluating the winners on {len(configs.VALIDATION_MAZES)} validation mazes ===", flush=True)
        cmd = [sys.executable, os.path.join(ROOT, "variants", "reevaluate_variant.py"), "--dir", d,
               "--mazes", *configs.VALIDATION_MAZES, "--seeds", str(a.seeds)]
        if a.workers:
            cmd += ["--workers", str(a.workers)]
        subprocess.run(cmd, check=True, cwd=ROOT)


def _mw(x, y):
    """Two-sided Mann-Whitney p (scipy if available) and A12 = P(x faster than y)."""
    a12 = sum((u < v) + 0.5 * (u == v) for u in x for v in y) / (len(x) * len(y))
    try:
        from scipy.stats import mannwhitneyu
        p = mannwhitneyu(x, y, alternative="two-sided").pvalue
    except Exception:
        p = float("nan")
    return p, a12


def _read(csv_path):
    """winners[label][scheme][maze] -> list of N90; baselines[scheme][maze] -> list; learned[label] -> [bool]"""
    winners = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    baselines = defaultdict(lambda: defaultdict(list))
    learned = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["maze"] not in configs.VALIDATION_MAZES:
                continue
            n = float(r["episodes_to_90"])
            if r["candidate"] == "baseline":
                baselines[r["scheme"]][r["maze"]].append(n)
            else:
                winners[r["candidate"]][r["scheme"]][r["maze"]].append(n)
                learned[r["candidate"]].append(r["converged"] == "True")
    return winners, baselines, learned


def _mean(v):
    return sum(v) / len(v)


def winner_score(per_maze, best_hand, score):
    """One number per winner. mean: mean N90 over the validation mazes (big mazes weigh more).
    relative: mean over the mazes of N90 / N90 of the best hand-made scheme on that maze
    (1.0 = as fast as the best hand-made scheme; every maze weighs the same)."""
    if score == "mean":
        return _mean([n for v in per_maze.values() for n in v])
    return _mean([_mean(v) / best_hand[m] for m, v in per_maze.items()])


def step_compare(a):
    rows, scores, hand = [], {}, defaultdict(list)
    ref = a.configs[0]
    for name in a.configs:
        d = folder(name)
        csv_path = os.path.join(d, "reevaluation.csv")
        if not os.path.exists(csv_path):
            print(f"{name}: no reevaluation.csv yet, skipped")
            continue
        winners, baselines, learned = _read(csv_path)
        best_hand = {m: min(_mean(b[m]) for b in baselines.values() if b.get(m)) for m in configs.VALIDATION_MAZES}
        for b, per in baselines.items():
            hand[b].append(winner_score(per, best_hand, a.score))
        runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "evolution", "*_run*.json")))]
        for label, algo in LABEL_ALGO.items():
            if label not in winners:
                continue
            sc = [winner_score(per, best_hand, a.score) for per in winners[label].values()]
            scores[(name, algo)] = sc
            rs = [r for r in runs if r["algo"] == algo]
            row = dict(config=name, **configs.CONFIGS[name], algo=ALGO_SHORT[algo], runs=len(sc),
                       budget="/".join(sorted({str(r["budget"]) for r in rs})),
                       evo_fitness=round(statistics.median(r["best"]["fitness"] for r in rs), 1),
                       score=a.score, validation=round(statistics.median(sc), 2),
                       validation_min=round(min(sc), 2), validation_max=round(max(sc), 2),
                       learned_pct=round(100 * _mean(learned[label])))
            for m in configs.VALIDATION_MAZES:       # median over winners of their mean N90 on this maze
                v = [_mean(per[m]) for per in winners[label].values() if per.get(m)]
                row[m] = round(statistics.median(v)) if v else ""
            if rs and "archive" in rs[0]:
                row["cells"] = statistics.median(len(r["archive"]) for r in rs)
                row["cells_learning"] = statistics.median(
                    sum(e["converged"] == len(e["n90"]) for e in r["archive"]) for r in rs)
            rows.append(row)
    if not rows:
        print("nothing to compare yet")
        return
    for row in rows:
        algo = LABEL_ALGO[f"{row['algo']} best"]
        if row["config"] != ref and (ref, algo) in scores:
            p, a12 = _mw(scores[(row["config"], algo)], scores[(ref, algo)])
            row["p_vs_ref"], row["A12_vs_ref"] = round(p, 3), round(a12, 2)

    what = ("mean episodes to 90% over the validation mazes" if a.score == "mean" else
            "episodes to 90% relative to the best hand-made scheme, averaged over the validation mazes (1 = as fast)")
    print(f"\nScreening on the validation mazes: {', '.join(configs.VALIDATION_MAZES)}")
    print(f"validation = {what};")
    print("one value per run winner, shown as median over runs [min-max]. learned = share of the re-evaluation")
    print(f"runs that reached 90%. Tests against '{ref}' (Mann-Whitney; A12 = chance to be faster, 0.5 = same).")
    print("With 3 runs per configuration only big differences can be significant: use the table to drop")
    print("clearly worse configurations and to pick candidates for full runs, not as a final result.\n")
    fmt = "{:15s}{:>5s}{:>5s}{:>8s}{:>9s}{:>22s}{:>9s}{:>9s}{:>6s}{:>7s}"
    print(fmt.format("config", "algo", "runs", "budget", "evo fit", "validation [min-max]", "learned", "p vs ref", "A12", "cells"))
    nd = 0 if a.score == "mean" else 2
    for row in sorted(rows, key=lambda r: (r["algo"], r["validation"])):
        val = f"{row['validation']:.{nd}f} [{row['validation_min']:.{nd}f}-{row['validation_max']:.{nd}f}]"
        print(fmt.format(row["config"], row["algo"], str(row["runs"]), row["budget"], f"{row['evo_fitness']:.0f}", val,
                         f"{row['learned_pct']}%", str(row.get("p_vs_ref", "")), str(row.get("A12_vs_ref", "")),
                         str(row.get("cells", ""))))
    print("\nper maze (median over winners of the mean episodes to 90%):")
    short = [m.replace("valid_", "") for m in configs.VALIDATION_MAZES]
    print(f"{'config':15s}{'algo':>5s}" + "".join(f"{m:>8s}" for m in short))
    for row in sorted(rows, key=lambda r: (r["algo"], r["validation"])):
        print(f"{row['config']:15s}{row['algo']:>5s}" + "".join(f"{str(row[m]):>8s}" for m in configs.VALIDATION_MAZES))
    if hand:
        print("\nbest hand-made schemes, same mazes and seeds, same score:")
        for b, v in sorted(hand.items(), key=lambda kv: _mean(kv[1]))[:5]:
            print(f"  {b:32s} {_mean(v):8.{nd}f}")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"comparison_{a.score}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {path}")


def step_list(a):
    print(f"{'config':15s}{'set':>8s}{'combine':>10s}{'gene ranges':>15s}{'cost':>6s}{'hours':>7s}")
    total = 0.0
    for name, c in configs.CONFIGS.items():
        h = configs.COST.get(c["set"], 1) * a.budget * a.runs * sum(SEC_PER_EVAL[x] for x in a.algos) * a.speed / 3600
        mark = " *" if name in a.configs else ""
        total += h if mark else 0
        print(f"{name:15s}{c['set']:>8s}{c['combine']:>10s}{c['variant']:>15s}{configs.COST.get(c['set'], 1):6.1f}{h:7.1f}{mark}")
    print(f"* = selected ({' '.join(a.configs)}): about {total:.1f} hours for {a.runs} runs x {len(a.algos)} algorithms"
          f" x budget {a.budget}")
    print("cost = time of one evaluation relative to single-maze. hours = laptop time (saved runs: 5000 evaluations")
    print("take ~17 min for MAP-Elites, ~5 min for DE), times --speed. Re-evaluation adds a few minutes per configuration.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["run", "reeval", "compare", "all", "list"])
    p.add_argument("--configs", nargs="+", default=configs.DEFAULT, choices=list(configs.CONFIGS),
                   help="the first one is the reference for the tests (default: " + " ".join(configs.DEFAULT) + ")")
    p.add_argument("--algos", nargs="+", default=["mapelites", "de"], choices=["mapelites", "de", "random"])
    p.add_argument("--runs", type=int, default=3, help="runs per configuration and algorithm (seeds 0..runs-1)")
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=10, help="re-evaluation seeds per maze")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="150 training episodes per run (code test only)")
    p.add_argument("--force", action="store_true", help="redo the re-evaluation even if it exists")
    p.add_argument("--score", default="mean", choices=["mean", "relative"],
                   help="compare step: mean N90 over the validation mazes, or relative to the best hand-made scheme")
    p.add_argument("--out", default=None, help="results folder (default results/screening)")
    p.add_argument("--speed", type=float, default=1.0, help="list step: time relative to the laptop (e.g. 0.25)")
    p.add_argument("--print-commands", action="store_true", help="run step: only print the commands")
    a = p.parse_args()
    global OUT
    OUT = os.path.abspath(a.out) if a.out else OUT
    if a.step == "list":
        step_list(a)
        return
    if a.step in ("run", "all"):
        step_run(a)
    if a.step in ("reeval", "all") and not a.print_commands:
        step_reeval(a)
    if a.step in ("compare", "all") and not a.print_commands:
        step_compare(a)


if __name__ == "__main__":
    main()
