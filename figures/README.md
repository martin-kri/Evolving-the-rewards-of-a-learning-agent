# Figures for the report

Extra figures, in their own folder like `variants/`. The scripts read the saved results and
the main code; none of the existing files is changed. Output goes to `results/figures/`
(ignored by git, like all of `results/`), as PNG (for slides) and PDF (vector, for LaTeX).

Run everything from the repository folder.

| script | figure | time |
|---|---|---|
| `fig_maps.py` | MAP-Elites map next to DE map, and map coverage over the run | seconds |
| `fig_maze_paths.py` | the routes the agents learn under different reward schemes | ~10 s |
| `fig_landscape.py` | fitness landscape of reward space (progress x step), winners on top | 2-10 min |
| `trace_run.py` | re-runs one evolution run and records every evaluation (needed for the DE map) | 5-20 min per run |

## 1. Maps: where each algorithm puts its reward schemes

```
python figures/trace_run.py --algo de --run 7     # once, ~5 min; reproduces the saved DE run 7 exactly
python figures/fig_maps.py
```

- (a) MAP-Elites archive of run 7: 63 cells, 54 of them learn on every seed.
- (b) DE run 7: every cell DE *evaluated* during the run (30), and its final population
  (1 cell, orange outline). DE keeps no map, so this is the most generous way to draw it.
- (c) Cells covered vs evaluations. MAP-Elites: all 10 runs. DE: every traced run.

Finding: DE found all of its 30 cells in its first 83 evaluations (essentially the random
start population). In the other 4,917 evaluations it found no new kind of reward scheme;
it only refined one. MAP-Elites keeps adding cells until the end of the budget.

To show all 10 DE runs in panel (c), trace the other runs (5 min each on the laptop):

```
0..9 | ForEach-Object { python figures/trace_run.py --algo de --run $_ }        # PowerShell
```

`--check results/2026-11-15_laptop/evolution/de_run7.json` confirms that a re-run is identical
to the saved run (it was, for run 7).

## 2. Routes on the maze

```
python figures/fig_maze_paths.py
python figures/fig_maze_paths.py --run 3 --elite-cell 4 0      # other run / other map cell
python figures/fig_maze_paths.py --maze maze_8x8                # another maze
```

Six panels: baselines A1, B1 and Russell & Norvig, the DE and MAP-Elites winners of run 7,
and the MAP-Elites elite of map cell (5, 0). Each scheme is trained on 20 new seeds (2000+);
line width = how many seeds' final greedy policy takes that step.

Finding: the training maze has two routes of the same length (16 safe steps), one above and
one below the wall block. The reward scheme decides which one the agent learns:

| scheme | learns in (median episodes) | upper / lower / fails |
|---|---|---|
| DE winner | 125 | 1 / 18 / 1 |
| MAP-Elites winner | 115 | 4 / 16 / 0 |
| MAP-Elites elite, cell (5, 0) | 105 | 13 / 6 / 1 |
| Russell & Norvig step cost | 250 | 13 / 5 / 2 |

Schemes with a strong progress reward take the lower route. The reason: the progress reward
measures distance to the goal *through* traps, so the middle trap looks like a shortcut and
the first step down from S already counts as progress. Equally fast schemes from different
map cells produce different behaviour, which is the point of keeping a diverse map.

`--shading` also shades the cells visited in the first 100 episodes (hardly differs here).

## 3. Fitness landscape

```
python figures/fig_landscape.py                       # progress x step, ~2 min (cloud) to ~5 min
python figures/fig_landscape.py --x trap --y wall     # another slice
python figures/fig_landscape.py --from-csv            # redraw only (after style changes)
```

Every cell is one reward scheme: progress and step from the axes, trap and wall fixed at the
median of the 20 winners (-6.87 and -1.76), delay 0. Mean episodes to learn on 10 new seeds
(3000+). The dashed box is the range the evolution searched. Winners are drawn by their
progress and step values; their trap and wall differ, so they are projected onto the slice.

Finding: the fast region is a diagonal band. A larger progress reward works best together
with a larger step cost. The DE winners sit at the right edge of the searched box (progress
= 1, its upper bound), where the band leaves the box; the MAP-Elites winners spread along
the band inside it. Fastest point inside the box: 100 episodes; outside, the best points
need 86 to 89 episodes (progress 1.5 to 2.5 with step -0.7 to -1.3; single grid points are
noisy, so read the band, not the single best point). This supports the
`wide_progress` / `wide_x2` variants and explains the variants screening. It also corrects
the one-gene sweep: a step cost below -1 only breaks learning when the progress reward stays
small.

## Using the figures in the report

- Use the PDFs with `\includegraphics`. The maze and map figures are made for the full page
  width (`figure*` in the IEEE two-column template); the landscape fits one column at
  `width=\columnwidth`.
- Colours are the same in every figure: MAP-Elites blue, DE orange, baselines grey; in the
  maps and the landscape darker grey = slower learning.
- Seeds: evolution 0-4, re-evaluation 1000-1029, maze figure 2000+, landscape 3000+, so no
  figure reuses the seeds a winner was selected on.

## Notes

- `fig_maze_paths.py` needs the final Q-table, which `qlearning.train()` does not return. The
  script wraps `qlearning.evaluate()` and the maze's `step()` while it runs to read it. If we
  keep this figure, a cleaner option is to let `train()` return the Q-table (one line).
- `trace_run.py` only wraps the function the algorithms call to evaluate genomes, so the run
  itself is unchanged. A trace file is about 1.3 MB.
- The MAP-Elites trace (`--algo mapelites`) is not needed for these figures. It would allow an
  animation of the map filling up, e.g. for a presentation.
