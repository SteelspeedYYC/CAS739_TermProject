# Interface for Generator
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class IGenerator(ABC):
    """
    Interface for generators.
    Each generator must output a 2D grid (numpy array) using encoding:
        0 = free
        1 = wall
        2 = checkpoint

    A generator may internally use a genome (bitstring, integer array, etc.)
    but must expose encode/decode + generate functions clearly.
    """
    def __init__(self, size: int):
        self._height = size
        self._width = size

    @abstractmethod
    def initialize_genome(self) -> Any:
        """
        Return a newly initialized genome representing a maze.
        The genome type is flexible (list, ndarray, dict, etc.)
        """
        pass

    @abstractmethod
    def decode(self, genome: Any) -> np.ndarray:
        """
        Decode genome → 2D numpy matrix of shape (H, W).
        Must include 0/1/2 according to global maze encoding.
        """
        pass

    @abstractmethod
    def mutate(self, genome: Any) -> Any:
        """
        Apply mutation and return a new genome.
        """
        pass

    @abstractmethod
    def crossover(self, g1: Any, g2: Any) -> Any:
        """
        Recombine two genomes and return a new offspring genome.
        """
        pass

    @abstractmethod
    def evaluate(self, grid: np.ndarray) -> float:
        """
        Evaluate the maze's standalone structural quality.
        NOTE: This is for baseline evaluation only — not solver fitness.
        (Used in Stage 1 to pre-fill MAP-Elites)
        """
        pass
