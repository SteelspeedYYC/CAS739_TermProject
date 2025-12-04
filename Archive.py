# Build the map class
from dataclasses import dataclass
from typing import Optional
import numpy as np
from Maze import Maze

@dataclass
class Individual:
    genome: np.ndarray                 # grid
    baseline_fitness: float
    solver_fitness: Optional[float]
    features: tuple[float, float]      # Normalized to [0,1]
    # f1 is Path Difficulty; f2 is Junction; ideally

class MapElitesArchive:
    """
    Simple 2D MAP-Elites Archive Over Two Features f1, f2 In [0, 1].
    """

    def __init__(self, bins_f1: int = 20, bins_f2: int = 20) -> None:
        self.bins_f1 = bins_f1
        self.bins_f2 = bins_f2

        # 2D Grid Of Optional Individuals
        self.grid: list[list[Optional[Individual]]] = [
            [None for _ in range(bins_f2)] for _ in range(bins_f1)
        ]


    def _feature_to_index(self, f1: float, f2: float) -> tuple[int, int]:
        """
        Map Continuous Features In [0, 1] x [0, 1] To Grid Indices.
        """
        f1_clamped = float(np.clip(f1, 0.0, 0.9999))
        f2_clamped = float(np.clip(f2, 0.0, 0.9999))

        i = int(f1_clamped * self.bins_f1)
        j = int(f2_clamped * self.bins_f2)
        return i, j


    def add_or_replace(self, indiv: Individual) -> None:
        """
        Place An Individual Into The Proper Cell.

        If The Cell Is Empty, Insert It.
        If There Is Already An Individual, Keep The One With Higher Baseline Fitness.
        """
        i, j = self._feature_to_index(*indiv.features)
        current = self.grid[i][j]

        if current is None or indiv.baseline_fitness > current.baseline_fitness:
            self.grid[i][j] = indiv

    def sample_parents(self, n: int, rng: np.random.Generator) -> list[Individual]:
        """
        Randomly Sample Up To n Individuals From Non-Empty Cells.
        """
        cells: list[Individual] = []

        # Collect All Non-Empty Cells
        for row in self.grid:
            for indiv in row:
                if indiv is not None:
                    cells.append(indiv)

        if not cells:
            return []

        # Randomly Choose Distinct Indices
        choose_n = min(n, len(cells))
        idx = rng.choice(len(cells), size=choose_n, replace=False)

        return [cells[int(k)] for k in idx]

    def iter_individuals(self) -> list[Individual]:
        """
        Return A List Of All Stored Individuals In The Archive.
        """
        cells: list[Individual] = []
        for row in self.grid:
            for indiv in row:
                if indiv is not None:
                    cells.append(indiv)
        return cells