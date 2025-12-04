import heapq
import numpy as np
from Maze.Maze import Maze
from .ISolver import ISolver


class AStarSolver(ISolver):
    def __init__(
        self,
        theta: np.ndarray | None = None,
        max_expansions: int = 10000,
    ) -> None:
        """
        theta: Parameter Vector Controlling The Heuristic.
        max_expansions: Hard Limit To Avoid Infinite Search.
        """
        if theta is None:
            # [d_goal, d_cp, wall_count, bias]
            theta = np.array([-1.0, -0.5, 0.2, 0.0], dtype=float)
        self.theta = theta.astype(float)
        self.max_expansions = int(max_expansions)

    # ---------------- Heuristic Definition ----------------

    def _features_for_state(
        self,
        maze: Maze,
        r: int,
        c: int,
        remaining_checkpoints: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Compute Feature Vector phi(s) For State s = (r, c, remaining_checkpoints).
        You Can Adjust This As You Like.
        """
        sr, sc = r, c
        gr, gc = maze.goal

        # Distance To Goal
        d_goal = abs(sr - gr) + abs(sc - gc)

        # Distance To Nearest Remaining Checkpoint (0 If None)
        if remaining_checkpoints:
            d_cp = min(
                abs(sr - cr) + abs(sc - cc)
                for (cr, cc) in remaining_checkpoints
            )
        else:
            d_cp = 0.0

        # Local Wall Density (4-Neighbour Walls)
        wall_count = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < maze.height and 0 <= nc < maze.width:
                if maze.grid[nr, nc] == 1:
                    wall_count += 1

        # Example Feature Vector: [d_goal, d_cp, wall_count, 1]
        return np.array([d_goal, d_cp, wall_count, 1.0], dtype=float)

    def _heuristic(
        self,
        maze: Maze,
        r: int,
        c: int,
        remaining_checkpoints: list[tuple[int, int]],
    ) -> float:
        phi = self._features_for_state(maze, r, c, remaining_checkpoints)
        d = min(len(self.theta), len(phi))
        return float(np.dot(self.theta[:d], phi[:d]))

    # ---------------- A* Search ----------------

    def solve(self, maze: Maze) -> tuple[bool, int]:
        checkpoints = maze._get_checkpoints()
        k = len(checkpoints)
        cp_index: dict[tuple[int, int], int] = {
            (r, c): i for i, (r, c) in enumerate(checkpoints)
        }
        full_mask = (1 << k) - 1

        sr, sc = maze.start
        gr, gc = maze.goal

        if maze.grid[sr, sc] == 1 or maze.grid[gr, gc] == 1:
            return False, 0

        start_mask = 0
        if k > 0 and (sr, sc) in cp_index:
            start_mask = 1 << cp_index[(sr, sc)]

        def key(rr: int, cc: int, mask: int) -> tuple[int, int, int]:
            return (rr, cc, mask)

        g_cost: dict[tuple[int, int, int], int] = {}
        open_heap: list[tuple[float, int, int, int, int]] = []

        start_state = key(sr, sc, start_mask)
        g_cost[start_state] = 0

        remaining_cps = self._remaining_checkpoints_from_mask(checkpoints, start_mask)
        h0 = self._heuristic(maze, sr, sc, remaining_cps)
        heapq.heappush(open_heap, (h0, 0, sr, sc, start_mask))

        expansions = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_heap and expansions < self.max_expansions:
            f, g, r, c, mask = heapq.heappop(open_heap)
            expansions += 1

            if (r, c) == (gr, gc) and mask == full_mask:
                return True, g

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < maze.height and 0 <= nc < maze.width:
                    if maze.grid[nr, nc] == 1:
                        continue

                    new_mask = mask
                    if k > 0 and maze.grid[nr, nc] == 2:
                        idx = cp_index.get((nr, nc))
                        if idx is not None:
                            new_mask = mask | (1 << idx)

                    ng = g + 1
                    state_k = key(nr, nc, new_mask)
                    old_g = g_cost.get(state_k)

                    if old_g is None or ng < old_g:
                        g_cost[state_k] = ng
                        remaining_cps = self._remaining_checkpoints_from_mask(
                            checkpoints, new_mask
                        )
                        h = self._heuristic(maze, nr, nc, remaining_cps)
                        heapq.heappush(open_heap, (ng + h, ng, nr, nc, new_mask))

        return False, self.max_expansions

    # Helper

    def _remaining_checkpoints_from_mask(
        self,
        checkpoints: list[tuple[int, int]],
        mask: int,
    ) -> list[tuple[int, int]]:
        """
        Return A List Of Remaining Checkpoints That Have Not Been Visited Yet.
        """
        remaining: list[tuple[int, int]] = []
        for i, (r, c) in enumerate(checkpoints):
            if not (mask & (1 << i)):
                remaining.append((r, c))
        return remaining
