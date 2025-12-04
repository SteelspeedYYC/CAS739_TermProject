# Interface for Solver
from abc import ABC, abstractmethod
from Maze.Maze import Maze


class ISolver(ABC):
    @abstractmethod
    def solve(self, maze: Maze) -> tuple[bool, int]:
        """
        Solve A Single Maze Instance.

        Returns:
            success: Whether A Valid Path Was Found (Visiting All Checkpoints)
            steps:   Number Of Steps Taken If Success, Or A Large Value If Not
        """
        pass