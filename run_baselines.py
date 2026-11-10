"""Run the baseline reward schemes on several mazes and summarise the results.

Examples (from inside the rl_project folder):
    python run_baselines.py --quick                     # fast check, about a minute
    python run_baselines.py                             # all mazes x all schemes x 10 seeds
    python run_baselines.py --mazes dyna_9x6_2traps dyna_9x6_3traps --seeds 20
    python run_baselines.py --random 3 --random-size 9x6 --random-traps 3
    python run_baselines.py --schemes A1_immediate_reward_only B1_immediate_reward_punish

Outputs (in --out, default "results"):
    runs.csv       one row per training run
    summary.csv    one row per maze x scheme
    overview.png   median episodes to 90% success, per maze and scheme
    curves_<maze>.png   mean learning curve of every scheme
    mazes.png      the maze layouts
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import statistics
import time

from gridmaze import LAYOUTS, get_maze, random_maze
from qlearning import DEFAULTS, train
from rewards import baseline_schemes

SUMMARY_FIELDS = ["maze", "scheme", "source", "shortest_path", "runs", "reached_90",
                  "median_episodes_to_90", "median_steps_to_90", "mean_final_success",
                  "mean_trap_falls", "mean_wall_bumps", "mean_punishment_share", "mean_feedback_delay"]


def run_one(job):
    maze, scheme, seed, settings = job
    return train(maze, scheme, seed, **settings)


def summarise(results):
    groups = {}
    for r in results:
        groups.setdefault((r["maze"], r["scheme"]), []).append(r)
    rows = []
    for (maze, scheme), rs in groups.items():
        reached = [r for r in rs if r["episodes_to_90"] is not None]
        rows.append(dict(
            maze=maze, scheme=scheme, source=rs[0]["source"], shortest_path=rs[0]["shortest_path"],
            runs=len(rs), reached_90=len(reached),
            median_episodes_to_90=statistics.median(r["episodes_to_90"] for r in reached) if reached else None,
            median_steps_to_90=statistics.median(r["steps_to_90"] for r in reached) if reached else None,
            mean_final_success=statistics.mean(r["final_success"] for r in rs),
            mean_trap_falls=statistics.mean(r["trap_falls"] for r in rs),
            mean_wall_bumps=statistics.mean(r["wall_bumps"] for r in rs),
            mean_punishment_share=statistics.mean(r["punishment_share"] for r in rs),
            mean_feedback_delay=statistics.mean(r["mean_feedback_delay"] for r in rs),
        ))
    return rows


def write_csvs(results, summary, out):
    run_fields = [k for k in results[0] if k not in ("curve", "eval_points")] + ["curve"]
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in run_fields if k != "curve"}
            row["curve"] = " ".join(f"{v:.2f}" for v in r["curve"])
            w.writerow(row)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary)


def print_summary(summary):
    head = f"{'maze':<18}{'scheme':<28}{'reached':>8}{'median ep':>10}{'final':>7}{'traps':>7}{'punish':>8}{'delay':>7}"
    print("\n" + head + "\n" + "-" * len(head))
    last = None
    for s in summary:
        if last and s["maze"] != last:
            print()
        last = s["maze"]
        med = "never" if s["median_episodes_to_90"] is None else f"{s['median_episodes_to_90']:.0f}"
        print(f"{s['maze']:<18}{s['scheme']:<28}{s['reached_90']:>4}/{s['runs']:<3}{med:>10}"
              f"{s['mean_final_success']:>7.2f}{s['mean_trap_falls']:>7.0f}"
              f"{s['mean_punishment_share']:>8.2f}{s['mean_feedback_delay']:>7.1f}")
    print("\nreached = runs that hit 90% test success; median ep = median episodes until then (over those runs);"
          "\nfinal = final test success; traps = trap falls during training; punish = share of negative feedback;"
          "\ndelay = mean steps between an event and its reward.")


def make_plots(results, summary, mazes, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib not found: skipping plots")
        return

    maze_names = [m.name for m in mazes]
    schemes = list(dict.fromkeys(s["scheme"] for s in summary))
    lookup = {(s["maze"], s["scheme"]): s for s in summary}

    # overview heatmap
    fig, ax = plt.subplots(figsize=(1.0 + 1.1 * len(maze_names), 0.9 + 0.42 * len(schemes)))
    vals = [[(lookup[(m, sc)]["median_episodes_to_90"] if (m, sc) in lookup else None)
             for m in maze_names] for sc in schemes]
    finite = [v for row in vals for v in row if v is not None]
    vmax = max(finite) if finite else 1
    grid = [[(v if v is not None else math.nan) for v in row] for row in vals]
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i, sc in enumerate(schemes):
        for j, m in enumerate(maze_names):
            s = lookup.get((m, sc))
            if not s:
                continue
            txt = "never" if s["median_episodes_to_90"] is None else f"{s['median_episodes_to_90']:.0f}"
            txt += f"\n{s['reached_90']}/{s['runs']}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="black" if s["median_episodes_to_90"] is None or s["median_episodes_to_90"] > vmax * 0.5 else "white")
    ax.set_xticks(range(len(maze_names)), maze_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(schemes)), schemes, fontsize=8)
    ax.set_title("Median episodes to 90% success (runs that got there / all runs)", fontsize=9)
    fig.colorbar(im, ax=ax, label="episodes (lower is faster)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "overview.png"), dpi=150)
    plt.close(fig)

    # learning curves per maze
    cmap20 = plt.get_cmap("tab20")
    for m in maze_names:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for k, sc in enumerate(schemes):
            rs = [r for r in results if r["maze"] == m and r["scheme"] == sc]
            if not rs:
                continue
            xs = rs[0]["eval_points"]
            mean = [statistics.mean(r["curve"][i] for r in rs) for i in range(len(xs))]
            ax.plot(xs, mean, label=sc, color=cmap20(k % 20), lw=1.6)
        ax.axhline(0.9, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("training episodes")
        ax.set_ylabel("test success rate (mean over seeds)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Learning curves: {m}")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"curves_{m}.png"), dpi=150)
        plt.close(fig)

    # maze layouts
    colours = ListedColormap(["white", "#555555", "#f08a6c", "#a7a1ea", "#5dcaa5"])
    code = {".": 0, "#": 1, "T": 2, "S": 3, "G": 4}
    cols = min(3, len(mazes))
    rows = math.ceil(len(mazes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), squeeze=False)
    for ax in list(axes.flat)[len(mazes):]:
        ax.axis("off")
    for ax, m in zip(axes.flat, mazes):
        ax.imshow([[code[ch] for ch in row] for row in m.rows], cmap=colours, vmin=0, vmax=4)
        for r, row in enumerate(m.rows):
            for c, ch in enumerate(row):
                if ch in "SG":
                    ax.text(c, r, ch, ha="center", va="center", fontsize=8)
        ax.set_xticks([x - 0.5 for x in range(m.W + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(m.H + 1)], minor=True)
        ax.grid(which="minor", color="#bbbbbb", lw=0.5)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_title(f"{m.name}\n{m.W}x{m.H}, shortest path {m.shortest_path}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mazes.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mazes", nargs="+", default=list(LAYOUTS), help=f"choose from: {', '.join(LAYOUTS)}")
    p.add_argument("--random", type=int, default=0, help="also add N random mazes")
    p.add_argument("--random-size", default="9x6", help="size of random mazes, WIDTHxHEIGHT")
    p.add_argument("--random-traps", type=int, default=3)
    p.add_argument("--random-walls", type=float, default=0.15, help="share of cells that are walls")
    p.add_argument("--schemes", nargs="+", default=None, help="scheme names (default: all baselines)")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--episodes", type=int, default=DEFAULTS["episodes"])
    p.add_argument("--slip", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="fast check: 2 mazes, 3 seeds, 300 episodes")
    args = p.parse_args()

    if args.quick:
        args.mazes, args.seeds, args.episodes = ["dyna_9x6_2traps", "dyna_9x6_3traps"], 3, 300

    mazes = [get_maze(name, slip=args.slip) for name in args.mazes]
    if args.random:
        w, h = (int(x) for x in args.random_size.lower().split("x"))
        for i in range(args.random):
            mazes.append(random_maze(h, w, wall_density=args.random_walls, n_traps=args.random_traps,
                                     seed=i, min_path=w + h - 2, slip=args.slip))

    settings = {"episodes": args.episodes}
    jobs = []
    for maze in mazes:
        schemes = baseline_schemes(maze)
        if args.schemes:
            unknown = set(args.schemes) - {s.name for s in schemes}
            if unknown:
                raise SystemExit(f"unknown scheme(s): {', '.join(sorted(unknown))}\n"
                                 f"available: {', '.join(s.name for s in schemes)}")
            schemes = [s for s in schemes if s.name in args.schemes]
        for scheme in schemes:
            for seed in range(args.seeds):
                jobs.append((maze, scheme, seed, settings))

    os.makedirs(args.out, exist_ok=True)
    print("Mazes:")
    for m in mazes:
        print("  ", m)
    print(f"{len(jobs)} training runs on {args.workers} worker(s)...")

    t0 = time.time()
    results = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=2), 1):
            results.append(r)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} runs done, {el:.0f}s elapsed, ~{el / i * (len(jobs) - i):.0f}s left")

    maze_rank = {m.name: a for a, m in enumerate(mazes)}
    scheme_rank = {s.name: k for k, s in enumerate(baseline_schemes(mazes[0]))}
    results.sort(key=lambda r: (maze_rank[r["maze"]], scheme_rank[r["scheme"]], r["seed"]))
    summary = summarise(results)
    summary.sort(key=lambda s: (maze_rank[s["maze"]], scheme_rank[s["scheme"]]))

    write_csvs(results, summary, args.out)
    print_summary(summary)
    make_plots(results, summary, mazes, args.out)
    print(f"\nSaved runs.csv, summary.csv and plots to '{args.out}' ({time.time() - t0:.0f}s total).")


if __name__ == "__main__":
    main()
