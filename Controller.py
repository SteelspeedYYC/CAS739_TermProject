# Controller for initialization, evolution, evaluation, and I/O to feature map.
import numpy as np

from Maze.Maze import Maze
from Maze.IGenerator import IGenerator
from Archive import MapElitesArchive, Individual

class ExperimentController:
    """
    Orchestrates Generation, Baseline Evaluation, And Map-Elites Loop.
    """

    def __init__(
        self,
        generator: IGenerator,
        archive: MapElitesArchive,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.generator = generator
        self.archive = archive
        self.rng = rng or np.random.default_rng()

    # Feature Computation 
    def _compute_features(self, maze: Maze, steps: float) -> tuple[float, float]:
        """
        Compute Two Features (f1, f2) Normalized To [0, 1]:

        f1 = Normalized Path Length (Difficulty)
        f2 = Normalized Openness (Structural Style)
        """
        # f1: Difficulty Based On Path Length
        max_path = (maze.height + maze.width) * 2.0  # Rough Upper Bound
        f1 = float(np.clip(steps / max_path, 0.0, 1.0))

        # f2: Structural Openness
        openness = maze._openness_score()
        # Assume Typical Openness Is In [0, 5]
        f2 = float(np.clip(openness / 5.0, 0.0, 1.0))

        return f1, f2


    def initialize_archive(self, n: int) -> None:
        """
        Generate n Random Mazes, Evaluate Baseline, Insert Into Archive.
        """
        for _ in range(n):
            genome = self.generator.initialize_genome()

            # Maze Handles Start/Goal Sampling
            maze = Maze(genome)

            fitness, steps = maze.evaluate_structure()

            f1, f2 = self._compute_features(maze, steps)

            indiv = Individual(
                genome=genome,
                baseline_fitness=fitness,
                solver_fitness=None,
                features=(f1, f2),
            )
            self.archive.add_or_replace(indiv)


    def step_evolution(self, offspring_per_step: int = 10) -> None:
        """
        One Map-Elites Evolution Step:

            1. Sample Parents From The Archive
            2. Generate Offspring Via Crossover + Mutation
            3. Evaluate Baseline
            4. Insert Into Archive
        """
        parents = self.archive.sample_parents(
            n=offspring_per_step * 2,
            rng=self.rng,
        )
        if not parents:
            return

        for _ in range(offspring_per_step):
            # Randomly Choose Two Parents
            if len(parents) >= 2:
                p1, p2 = self.rng.choice(parents, size=2, replace=False)
                child_genome = self.generator.crossover(p1.genome, p2.genome)
            else:
                # If Only One Parent Is Available, Use Mutation Only
                p1 = parents[0]
                child_genome = p1.genome.copy()

            # Apply Mutation
            child_genome = self.generator.mutate(child_genome)

            # Evaluate Offspring
            maze = Maze(child_genome)
            fitness, steps = maze.evaluate_structure()

            f1, f2 = self._compute_features(maze, steps)

            child = Individual(
                genome=child_genome,
                baseline_fitness=fitness,
                solver_fitness=None,
                features=(f1, f2),
            )
            self.archive.add_or_replace(child)

    # Solver Interaction

    def sample_envs_for_solver(self, k: int) -> list[Individual]:
        """
        Select Environments For Solver Training.

        For Now:
            Simply Take Top-k Individuals By Baseline Fitness.
            Later You Can Mix In Feature-Based Or Diversity-Based Selection.
        """
        all_indiv = self.archive.iter_individuals()
        if not all_indiv:
            return []

        # Sort By Baseline Fitness Descending
        all_indiv.sort(key=lambda ind: ind.baseline_fitness, reverse=True)
        return all_indiv[:k]
