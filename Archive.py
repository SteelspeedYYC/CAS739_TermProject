# Build the map class
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from .Maze import Maze

@dataclass
class Individual:
    genome: np.ndarray                 # grid
    baseline_score: float
    solver_score: Optional[float]
    features: Tuple[float, float]      # Normalized to [0,1]


class MapElitesArchive:
    """
    Simple 2D MAP-Elites Archive.
    Feature Space: (f1, f2) In [0, 1] x [0, 1] (You Can Normalize)
    """

    def __init__(self, bins_f1: int = 20, bins_f2: int = 20) -> None:
        self.BinsF1 = bins_f1
        self.BinsF2 = bins_f2

        # Archive Is A 2D Grid Of Optional Individuals
        self.Grid: list[list[Optional[Individual]]] = [
            [None for _ in range(bins_f2)] for _ in range(bins_f1)
        ]

    def _feature_to_index(self, f1: float, f2: float) -> Tuple[int, int]:
        """
        Map Continuous Features In [0, 1] To Grid Indices.
        """
        i = int(np.clip(f1, 0.0, 0.9999) * self.BinsF1)
        j = int(np.clip(f2, 0.0, 0.9999) * self.BinsF2)
        return i, j

    def add_or_replace(self, indiv: Individual) -> None:
        i, j = self._feature_to_index(*indiv.features)
        current = self.Grid[i][j]
        # Replace If Empty Or New Baseline Score Is Better
        if current is None or indiv.baseline_score > current.baseline_score:
            self.Grid[i][j] = indiv

    def sample_parents(self, n: int, rng: np.random.Generator) -> list[Individual]:
        """
        Randomly Sample Up To n Individuals From Non-Empty Cells.
        """
        cells: list[Individual] = []
        for row in self.Grid:
            for indiv in row:
                if indiv is not None:
                    cells.append(indiv)

        if not cells:
            return []

        idx = rng.choice(len(cells), size=min(n, len(cells)), replace=False)
        return [cells[int(k)] for k in idx]