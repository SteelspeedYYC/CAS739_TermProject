import heapq
import numpy as np
from Maze.Maze import Maze
from .ISolver import ISolver
from collections import deque


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


class GreedySolver(ISolver):
    """
    Greedy Maze Solver:

    At Each Step, Look At Neighbour States And Choose The One
    With The Lowest Heuristic Value h_theta(s').
    No Global Search, Only Local Decisions.
    Modified with AI advice
    """

    def __init__(
        self,
        theta: np.ndarray,
        max_steps_factor: float = 4.0,
        memory_length: int = 20,  # size of short-term memory
        repeat_penalty: float = 100.0, # penalty for revisiting
    ) -> None:
        """
        theta: Parameter Vector For The Heuristic h_theta(s).
        max_steps_factor: Max Steps = factor * (H * W) Per Episode.
        memory_length: How many recent steps to remember to avoid loops.
        repeat_penalty: Heuristic penalty for stepping on a recently visited cell.
        """
        self.theta = theta.astype(float)
        self.max_steps_factor = float(max_steps_factor)
        self.memory_length = int(memory_length)
        self.repeat_penalty = float(repeat_penalty)

    # Feature / Heuristic

    def _features_for_state(
        self,
        maze: Maze,
        r: int,
        c: int,
        remaining_checkpoints: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Feature Vector phi(s) For Local Heuristic.
        """
        sr, sc = r, c
        gr, gc = maze.goal

        # Distance To Goal
        d_goal = abs(sr - gr) + abs(sc - gc)

        # Distance To Nearest Remaining Checkpoint
        if remaining_checkpoints:
            d_cp = min(
                abs(sr - cr) + abs(sc - cc)
                for (cr, cc) in remaining_checkpoints
            )
        else:
            d_cp = 0.0

        # Local Wall Count (4-Neighbour)
        wall_count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < maze.height and 0 <= nc < maze.width:
                if maze.grid[nr, nc] == 1:
                    wall_count += 1

        # Bias Term 1.0 At The End
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

    def _remaining_checkpoints_from_mask(
        self,
        checkpoints: list[tuple[int, int]],
        mask: int,
    ) -> list[tuple[int, int]]:
        remaining: list[tuple[int, int]] = []
        for i, (r, c) in enumerate(checkpoints):
            if not (mask & (1 << i)):
                remaining.append((r, c))
        return remaining


    # Greedy Solve
    # NO MORE USE!!!
    def solve(self, maze: Maze) -> tuple[bool, int]:
        """
        Greedy Walk:

            State = (Row, Col, VisitedMask)
            At Each Step:
                - Look At Up To 4 Neighbours
                - Compute h_theta For Each
                - Move To The Neighbour With Lowest h
        """
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

        # Max Steps Based On Maze Size
        max_steps = int(self.max_steps_factor * maze.height * maze.width)
        if max_steps <= 0:
            max_steps = maze.height * maze.width

        r, c = sr, sc
        mask = start_mask
        steps = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        min_goal_dist = abs(sr - gr) + abs(sc - gc)
        visited_cp_count = 0

        prev_pos: tuple[int, int] | None = None
        backtrack_penalty = 5.0
        visited_cp_count = bin(start_mask).count("1")

        while steps < max_steps:
            # Check Goal Condition
            if (r, c) == (gr, gc):
                total_cp = len(checkpoints)
                if total_cp > 0:
                    cp_ratio = visited_cp_count / float(total_cp)
                    bonus = int(cp_ratio * 0.5 * steps)
                    eff_steps = max(1, steps - bonus)
                else:
                    eff_steps = steps

                return True, eff_steps

            # Collect Candidate Moves
            best_h = None
            best_state: tuple[int, int, int] | None = None

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < maze.height and 0 <= nc < maze.width:
                    if maze.grid[nr, nc] == 1:
                        continue  # Wall

                    new_mask = mask
                    if k > 0 and maze.grid[nr, nc] == 2:
                        idx = cp_index.get((nr, nc))
                        if idx is not None:
                            new_mask = mask | (1 << idx)

                    remaining = self._remaining_checkpoints_from_mask(
                        checkpoints, new_mask
                    )
                    h = self._heuristic(maze, nr, nc, remaining)

                    # punish swing
                    if prev_pos is not None and (nr, nc) == prev_pos:
                        h += backtrack_penalty

                    if best_h is None or h < best_h:
                        best_h = h
                        best_state = (nr, nc, new_mask)

            if best_state is None:
                # Dead End
                break

            # Move Greedily To Best Neighbour
            prev_pos = (r, c)          # update last position
            r, c, mask = best_state
            steps += 1

            visited_cp_count = max(
                visited_cp_count,
                bin(mask).count("1"),
            )
            # Update Progress Metrics
            d_goal = abs(r - gr) + abs(c - gc)
            if d_goal < min_goal_dist:
                min_goal_dist = d_goal


        # Failed To Find Valid Path Within Step Limit
        total_cp = len(checkpoints)
        if total_cp > 0:
            cp_ratio = visited_cp_count / float(total_cp)
        else:
            cp_ratio = 0.0

        max_d_goal = maze.height + maze.width
        if max_d_goal > 0:
            goal_progress = (max_d_goal - float(min_goal_dist)) / float(max_d_goal)
        else:
            goal_progress = 0.0

        progress = 0.7 * cp_ratio + 0.3 * goal_progress
        progress *= 1.5
        progress = float(np.clip(progress, 0.0, 1.0))

        effective_steps = int((1.0 - progress) * max_steps)
        effective_steps = max(1, min(max_steps, effective_steps))

        return False, effective_steps


    # Add another solve func, can also return cp pass ratio
    def solve_with_stats(self, maze: Maze) -> tuple[bool, int, float]:
        """
        Modified solve method with Tabu List (Short-Term Memory) to prevent oscillation.
        """
        checkpoints = maze._get_checkpoints()
        k = len(checkpoints)
        cp_index: dict[tuple[int, int], int] = {
            (r, c): i for i, (r, c) in enumerate(checkpoints)
        }

        sr, sc = maze.start
        gr, gc = maze.goal

        if maze.grid[sr, sc] == 1 or maze.grid[gr, gc] == 1:
            return False, 0, 0.0

        start_mask = 0
        if k > 0 and (sr, sc) in cp_index:
            start_mask = 1 << cp_index[(sr, sc)]

        max_steps = int(self.max_steps_factor * maze.height * maze.width)
        if max_steps <= 0:
            max_steps = maze.height * maze.width

        r, c = sr, sc
        mask = start_mask
        steps = 0

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited_cp_count = bin(start_mask).count("1")
        
        # Stores (r, c) of the last N steps
        history = deque(maxlen=self.memory_length)
        history.append((r, c))

        while steps < max_steps:
            if (r, c) == (gr, gc):
                total_cp = len(checkpoints)
                cp_ratio = (
                    visited_cp_count / float(total_cp)
                    if total_cp > 0 else 0.0
                )
                return True, steps, cp_ratio

            best_h = None
            best_state: tuple[int, int, int] | None = None

            # 1. Collect valid moves
            candidates = []
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

                    remaining = self._remaining_checkpoints_from_mask(
                        checkpoints, new_mask
                    )
                    
                    # Base heuristic
                    h = self._heuristic(maze, nr, nc, remaining)

                    # If (nr, nc) is in our short-term memory, add a huge penalty.
                    # This forces the solver to explore new cells instead of oscillating.
                    if (nr, nc) in history:
                        # Count how many times we visited it recently (optional, or just static penalty)
                        visits = history.count((nr, nc))
                        h += self.repeat_penalty * visits

                    candidates.append((h, nr, nc, new_mask))

            # 2. Pick best move
            if not candidates:
                break # Dead end with no moves

            # Sort by h (lowest is best)
            # Add small random noise to break ties if h is identical
            candidates.sort(key=lambda x: x[0] + np.random.uniform(0, 1e-6))
            
            best_h, next_r, next_c, next_mask = candidates[0]
            
            # Update state
            r, c, mask = next_r, next_c, next_mask
            steps += 1
            
            # Update memory
            history.append((r, c))

            visited_cp_count = max(
                visited_cp_count,
                bin(mask).count("1"),
            )

        total_cp = len(checkpoints)
        cp_ratio = (
            visited_cp_count / float(total_cp)
            if total_cp > 0 else 0.0
        )
        return False, steps, cp_ratio