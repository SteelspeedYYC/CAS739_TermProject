from .IGenerator import IGenerator

from typing import Any
import numpy as np

class PerlinMazeGenerator(IGenerator):
    """
    Maze generator based on smoothed noise (Perlin-like).

    Genome representation:
        - A 2D numpy array of shape (size, size) with values in {0, 1, 2}
        - 0 = free cell
        - 1 = wall
        - 2 = checkpoint

    initialize_genome() uses smoothed random noise to create
    more "organic" wall patterns instead of pure random grids.
    """

    def __init__(
        self,
        size: int,
        # Threshold for wall generation
        wall_threshold: float = 0.4,
        # Threshold for checkpoint generation
        checkpoint_low: float = 0.65,
        checkpoint_high: float = 0.8,
        # Noise parameter
        smooth_steps: int = 3,
        # Get consistent result with same seeds
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(size)

        self.wall_threshold = wall_threshold
        self.checkpoint_low = checkpoint_low
        self.checkpoint_high = checkpoint_high
        self.smooth_steps = smooth_steps
        self.rng = rng or np.random.default_rng()


    # Internal helper func
    def _perlin_like_noise(self) -> np.ndarray:
        """
        Generate a smooth noise field using repeated 3x3 smoothing.
        Not a strict Perlin implementation, but visually similar enough
        for maze structure.
        """
        h, w = self._height, self._width
        a = self.rng.random((h, w))

        for _ in range(self.smooth_steps):
            # Use: 3x3 kernel: [1 2 1; 2 4 2; 1 2 1] / 16
            center = 4 * a
            plus = (
                np.roll(a, 1, axis=0) +
                np.roll(a, -1, axis=0) +
                np.roll(a, 1, axis=1) +
                np.roll(a, -1, axis=1)
            )
            diag = (
                np.roll(np.roll(a, 1, axis=0), 1, axis=1) +
                np.roll(np.roll(a, 1, axis=0), -1, axis=1) +
                np.roll(np.roll(a, -1, axis=0), 1, axis=1) +
                np.roll(np.roll(a, -1, axis=0), -1, axis=1)
            )
            a = (center + 2 * plus + diag) / 16.0

        # Normalization
        a_min, a_max = a.min(), a.max()
        if a_max > a_min:
            a = (a - a_min) / (a_max - a_min)
        else:
            a = np.zeros_like(a)
        return a
    

    def initialize_genome(self) -> np.ndarray:
        """
        Create a maze genome using smoothed noise:
            - values below wall_threshold become walls (1)
            - values above threshold are free (0)
            - a mid–high band becomes checkpoints (2)
        """
        h, w = self._height, self._width
        noise = self._perlin_like_noise()

        grid = np.zeros((h, w), dtype=np.int8)

        # Walls
        grid[noise < self.wall_threshold] = 1

        # Checkpoints in a band of higher noise values, only on free cells
        cp_mask = (
            (noise >= self.checkpoint_low) &
            (noise < self.checkpoint_high) &
            (grid == 0)
        )
        grid[cp_mask] = 2

        # Checkpoint checking, must be at least one cp.
        if not (grid == 2).any():
            r_idx = self.rng.integers(0, h)
            c_idx = self.rng.integers(0, w)
            grid[r_idx, c_idx] = 2

        return grid

    def decode(self, genome: Any) -> np.ndarray:
        """
        For now genome is already the grid; just clone it.
        """
        return np.asarray(genome, dtype=np.int8).copy()

    def mutate(self, genome: Any) -> np.ndarray:
        """
        Simple cell-level mutation within this Perlin-style grid.
        """
        child = np.asarray(genome, dtype=np.int8).copy()
        h, w = child.shape
        total = h * w

        num_mut = max(1, int(0.01 * total))
        indices = self.rng.choice(total, size=num_mut, replace=False)

        flat = child.ravel()
        for idx in indices:
            v = flat[idx]
            if v == 0:
                flat[idx] = 1
            elif v == 1:
                flat[idx] = 0
            else:  # v == 2, checkpoint
                if self.rng.random() < 0.5:
                    flat[idx] = 0

        return child.reshape(h, w)

    def crossover(self, g1: Any, g2: Any) -> np.ndarray:
        """
        Same-type crossover: uniform blending of two Perlin-style mazes.
        """
        g1 = np.asarray(g1, dtype=np.int8)
        g2 = np.asarray(g2, dtype=np.int8)
        assert g1.shape == g2.shape, "Genomes must have the same shape for crossover."

        mask = self.rng.random(g1.shape) < 0.5
        child = np.where(mask, g1, g2)
        return child.astype(np.int8)

    def evaluate(self, grid: Any) -> float:
        grid = np.asarray(grid, dtype=np.int8)
        free_ratio = np.mean(grid == 0)
        wall_ratio = np.mean(grid == 1)
        checkpoint_count = int(np.sum(grid == 2))

        score = free_ratio - 0.3 * wall_ratio + 0.05 * checkpoint_count
        return float(score)
    

class DFSMazeGenerator(IGenerator):
    """
    Maze generator based on DFS. A different Algo baseline here for different geno.

    Genome representation:
        - A 2D numpy array of shape (size, size) with values in {0, 1, 2}
        - 0 = free cell
        - 1 = wall
        - 2 = checkpoint

    initialize_genome() uses smoothed random noise to create
    more "organic" wall patterns instead of pure random grids. Same as Perlin Generator.
    """
    def __init__(
        self,
        size: int,
        checkpoint_ratio: float = 0.02,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Args:
            size:
                Maze Size (Size x Size). Same As IGenerator Base Class.
            checkpoint_ratio:
                Approximate Ratio Of Corridor Cells That Will Be Marked As Checkpoints.
            rng:
                Optional Random Generator For Reproducibility.
        """
        super().__init__(size)
        self.CheckpointRatio = checkpoint_ratio
        self.Rng = rng or np.random.default_rng()

    # Internal Helper: DFS-Based Carving On An Implicit Cell Grid
    def _dfs_carve(self) -> np.ndarray:
        size = self._height
        grid = np.ones((size, size), dtype=np.int8)  # Start With All Walls

        # List All Cell Coordinates (Odd, Odd)
        cell_rows = list(range(1, size, 2))
        cell_cols = list(range(1, size, 2))
        if not cell_rows or not cell_cols:
            # Very Small Size Fallback: Just Return All Free
            return np.zeros((size, size), dtype=np.int8)

        # Visited Set For Cells
        visited = set()

        # Choose Random Start Cell
        start_r = int(self.Rng.choice(cell_rows))
        start_c = int(self.Rng.choice(cell_cols))
        stack: list[tuple[int, int]] = [(start_r, start_c)]
        visited.add((start_r, start_c))

        # Mark Start Cell As Free
        grid[start_r, start_c] = 0

        # DFS Backtracking
        while stack:
            r, c = stack[-1]

            # Collect Unvisited Neighbors Two Steps Away
            neighbors: list[tuple[int, int]] = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < size - 1 and 1 <= nc < size - 1:
                    if (nr, nc) not in visited:
                        neighbors.append((nr, nc))

            if not neighbors:
                # No Unvisited Neighbor, Backtrack
                stack.pop()
                continue

            # Choose One Neighbor Randomly
            nr, nc = neighbors[int(self.Rng.integers(0, len(neighbors)))]
            visited.add((nr, nc))
            stack.append((nr, nc))

            # Carve Passage Between (r, c) And (nr, nc)
            wall_r = (r + nr) // 2
            wall_c = (c + nc) // 2
            grid[wall_r, wall_c] = 0
            grid[nr, nc] = 0

        # Optionally: Ensure Outer Border Is Walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        return grid

    def initialize_genome(self) -> np.ndarray:
        """
        Generate One DFS-Style Maze Genome:
        First Carve Corridors Using DFS.
        Then Add Checkpoints On A Subset Of Free Cells.
        """
        grid = self._dfs_carve()

        # Collect All Free Cells (Candidates For Checkpoints)
        free_positions = np.argwhere(grid == 0)
        if free_positions.size > 0:
            num_free = free_positions.shape[0]
            num_cp = max(1, int(self.CheckpointRatio * num_free))

            # Sample Without Replacement
            indices = self.Rng.choice(num_free, size=num_cp, replace=False)
            for idx in indices:
                r, c = free_positions[idx]
                grid[r, c] = 2

        return grid

    def decode(self, genome: Any) -> np.ndarray:
        return np.asarray(genome, dtype=np.int8).copy()

    def mutate(self, genome: Any) -> np.ndarray:
        """
        Simple Cell-Level Mutation On DFS Maze:
        Flip Some Walls To Free And Vice Versa.
        Occasionally Remove Checkpoints.
        """
        child = np.asarray(genome, dtype=np.int8).copy()
        h, w = child.shape
        total = h * w

        # Mutate About 1% Of Cells
        num_mut = max(1, int(0.01 * total))
        indices = self.Rng.choice(total, size=num_mut, replace=False)

        flat = child.ravel()
        for idx in indices:
            v = flat[idx]
            if v == 0:
                flat[idx] = 1      # Free -> Wall
            elif v == 1:
                flat[idx] = 0      # Wall -> Free
            else:
                # Checkpoint -> Free With Some Probability
                if self.Rng.random() < 0.5:
                    flat[idx] = 0

        return child.reshape(h, w)

    def crossover(self, g1: Any, g2: Any) -> np.ndarray:
        """
        Uniform Crossover Between Two DFS Mazes:
        Each Cell Chooses From Parent 1 Or Parent 2 With Probability 0.5.
        """
        g1 = np.asarray(g1, dtype=np.int8)
        g2 = np.asarray(g2, dtype=np.int8)
        assert g1.shape == g2.shape, "Genomes Must Have The Same Shape For Crossover."

        mask = self.Rng.random(g1.shape) < 0.5
        child = np.where(mask, g1, g2)
        return child.astype(np.int8)

    def evaluate(self, grid: Any) -> float:
        grid = np.asarray(grid, dtype=np.int8)
        free_ratio = np.mean(grid == 0)
        wall_ratio = np.mean(grid == 1)
        checkpoint_count = int(np.sum(grid == 2))

        # Simple Heuristic: Prefer Some Free Space, Not Too Many Walls, Some Checkpoints
        score = free_ratio - 0.3 * wall_ratio + 0.05 * checkpoint_count
        return float(score)