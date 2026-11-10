"""Grid maze environment for the reward-scheme experiments.

A maze is written as text, one string per row:
    '.' free cell      '#' wall      'T' trap
    'S' start          'G' goal

Actions: 0 = up, 1 = down, 2 = left, 3 = right.
With probability `slip` the agent moves in a random direction instead.
Moving into a wall or off the grid leaves the agent where it is (a "wall bump").
Reaching the goal or stepping into a trap ends the episode.
"""
from collections import deque
import random

ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Maze:
    def __init__(self, name, rows, slip=0.1):
        self.name = name
        self.rows = list(rows)
        self.H = len(rows)
        self.W = len(rows[0])
        if any(len(r) != self.W for r in rows):
            raise ValueError(f"{name}: all rows must have the same length")
        self.slip = slip

        self.walls, self.traps = set(), set()
        start = goal = None
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "T":
                    self.traps.add((r, c))
                elif ch == "S":
                    start = (r, c)
                elif ch == "G":
                    goal = (r, c)
                elif ch != ".":
                    raise ValueError(f"{name}: unknown character {ch!r}")
        if start is None or goal is None:
            raise ValueError(f"{name}: maze needs one 'S' and one 'G'")

        self.n_states = self.H * self.W
        self.start = self._id(*start)
        self.goal = self._id(*goal)
        self.is_trap = [False] * self.n_states
        for r, c in self.traps:
            self.is_trap[self._id(r, c)] = True

        # next_state[s][a]: where action a leads from state s (stays put on a bump)
        self.next_state = [[s] * 4 for s in range(self.n_states)]
        for r in range(self.H):
            for c in range(self.W):
                s = self._id(r, c)
                for a, (dr, dc) in enumerate(ACTIONS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.H and 0 <= nc < self.W and (nr, nc) not in self.walls:
                        self.next_state[s][a] = self._id(nr, nc)

        # distance to the goal: traps passable (used by the progress reward)
        self.dist = self._bfs(block_traps=False)
        # distance to the goal: traps blocked (the safe shortest path)
        self.safe_dist = self._bfs(block_traps=True)
        self.shortest_path = self.safe_dist[self.start]
        if self.shortest_path is None:
            raise ValueError(f"{name}: no safe path from S to G")
        # cells that cannot reach the goal at all get a large distance
        far = max(d for d in self.dist if d is not None) + 1
        self.dist = [far if d is None else d for d in self.dist]

    def _id(self, r, c):
        return r * self.W + c

    def _bfs(self, block_traps):
        dist = [None] * self.n_states
        dist[self.goal] = 0
        queue = deque([self.goal])
        while queue:
            s = queue.popleft()
            r, c = divmod(s, self.W)
            for dr, dc in ACTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.H and 0 <= nc < self.W):
                    continue
                if (nr, nc) in self.walls or (block_traps and (nr, nc) in self.traps):
                    continue
                n = self._id(nr, nc)
                if dist[n] is None:
                    dist[n] = dist[s] + 1
                    queue.append(n)
        return dist

    def step(self, s, a, rng):
        """One move. Returns (next_state, bumped, reached_goal, fell_in_trap)."""
        if rng.random() < self.slip:
            a = rng.randrange(4)
        ns = self.next_state[s][a]
        return ns, ns == s, ns == self.goal, self.is_trap[ns]

    def __repr__(self):
        return (f"Maze({self.name}: {self.W}x{self.H}, {len(self.walls)} walls, "
                f"{len(self.traps)} traps, shortest path {self.shortest_path})")


# ---------------------------------------------------------------- layouts
LAYOUTS = {
    # DeepMind AI Safety Gridworlds, lava world (training version), inside of the walls
    "lava_7x5": [
        "S.TTT.G",
        ".......",
        ".......",
        ".......",
        "..TTT..",
    ],
    # Sutton & Barto, Dyna Maze (Example 8.1), original
    "dyna_9x6": [
        ".......#G",
        "..#....#.",
        "S.#....#.",
        "..#......",
        ".....#...",
        ".........",
    ],
    # Dyna Maze with 2 traps added (our candidate)
    "dyna_9x6_2traps": [
        ".......#G",
        "..#....#.",
        "S.#....#.",
        "..#..T...",
        ".....#...",
        "..T......",
    ],
    # Dyna Maze with 3 traps added (our candidate)
    "dyna_9x6_3traps": [
        ".....T.#G",
        "..#....#.",
        "S.#....#.",
        "..#..T...",
        ".....#...",
        "..T......",
    ],
    "maze_8x8": [
        ".......G",
        "....T...",
        ".#####..",
        "...T....",
        ".T......",
        "..#####.",
        ".....T..",
        "S.......",
    ],
    "maze_10x10": [
        ".........G",
        ".....T....",
        "........T.",
        "########..",
        ".T........",
        "......T...",
        "..########",
        "..........",
        "....T..T..",
        "S.........",
    ],
}


def get_maze(name, slip=0.1):
    return Maze(name, LAYOUTS[name], slip=slip)


def random_maze(rows, cols, wall_density=0.2, n_traps=3, seed=0,
                min_path=None, slip=0.1, max_tries=10000):
    """Random obstacle maze: start bottom-left, goal top-right, scattered walls and traps.

    Keeps drawing until there is a safe path (and, if given, one of at least `min_path` steps).
    """
    rng = random.Random(seed)
    start, goal = (rows - 1, 0), (0, cols - 1)
    free = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in (start, goal)]
    n_walls = int(round(wall_density * rows * cols))
    for _ in range(max_tries):
        rng.shuffle(free)
        grid = [["."] * cols for _ in range(rows)]
        for r, c in free[:n_walls]:
            grid[r][c] = "#"
        for r, c in free[n_walls:n_walls + n_traps]:
            grid[r][c] = "T"
        grid[start[0]][start[1]] = "S"
        grid[goal[0]][goal[1]] = "G"
        try:
            maze = Maze(f"random_{cols}x{rows}_s{seed}", ["".join(r) for r in grid], slip=slip)
        except ValueError:
            continue
        if min_path is None or maze.shortest_path >= min_path:
            return maze
    raise RuntimeError("could not generate a solvable maze; lower wall_density or n_traps")
