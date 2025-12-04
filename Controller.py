# Controller for initialization, evolution, evaluation, and I/O to feature map.
import numpy as np

from Maze.Maze import Maze
from Maze.IGenerator import IGenerator
from Archive import MapElitesArchive, Individual
import MazeSolver.Solver as Solver
import MazeSolver.SolverPopulation as SolverPop


class ExperimentController:
    """
    Orchestrates Generation, Baseline Evaluation, Map-Elites Loop,
    And Co-Evolution Of Maze Generator And Solver.
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

        # ES Population Of Solvers (Greedy With Different Theta)
        self.solver_population: list[SolverPop.SolverIndividual] = []


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

 
    # Archive Initialization And Maze Evolution
    def initialize_archive(self, n: int) -> None:
        """
        Generate n Random Mazes, Evaluate Baseline, Insert Into Archive.
        (Single Generator Version, Mostly For Quick Tests.)
        """
        for _ in range(n):
            genome = self.generator.initialize_genome()

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

    def initialize_archive_mixed(
        self,
        n: int,
        perlin_gen: IGenerator,
        dfs_gen: IGenerator,
    ) -> None:
        """
        Initialize Archive With A 50-50 Mix Of Perlin And DFS Mazes.

        n: Total Number Of Seeds (Approx Half Perlin, Half DFS).
        This Function Skips Degenerate Mazes That Have Fewer Than
        Two Free Cells (So Start/Goal Sampling Does Not Fail).
        """
        added = 0
        max_tries = n * 10
        tries = 0

        while added < n and tries < max_tries:
            tries += 1

            # Alternate Between Perlin / DFS
            if added % 2 == 0:
                gen = perlin_gen
            else:
                gen = dfs_gen

            genome = gen.initialize_genome()

            # Free Cell Checking
            num_free = int(np.count_nonzero(genome == 0))
            if num_free < 2:
                continue

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
            added += 1

        if added < n:
            print(f"Warning: Only Generated {added} Valid Mazes (Requested {n}).")

    def _sample_parents_by_solver_fitness(self, n: int) -> list[Individual]:
        """
        Sample Parents Based On Solver Fitness (Higher Is Better).

        Only Individuals With Non-None solver_fitness Are Considered.
        """
        all_indiv = self.archive.iter_individuals()
        scored = [ind for ind in all_indiv if ind.solver_fitness is not None]
        if not scored:
            return []

        if len(scored) <= n:
            return scored

        idx = self.rng.choice(len(scored), size=n, replace=False)
        return [scored[int(i)] for i in idx]

    def step_evolution(
        self,
        offspring_per_step: int = 10,
        use_solver_fitness: bool = False,
    ) -> None:
        """
        One Map-Elites Evolution Step:

            1. Sample Parents From The Archive
               - Either Uniformly (baseline) Or Biased By Solver Fitness
            2. Generate Offspring Via Crossover + Mutation
            3. Evaluate Baseline
            4. Insert Into Archive
        """
        if use_solver_fitness:
            parents = self._sample_parents_by_solver_fitness(
                n=offspring_per_step * 2
            )
        else:
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

    # Solver Interaction (Sampling Envs, ES Evolution, Writing solver_fitness)
    def sample_envs_for_solver(self, k: int) -> list[Individual]:
        """
        Select Environments For Solver Training.
        Currently: Take Top-k Individuals By Baseline Fitness.
        """
        all_indiv = self.archive.iter_individuals()
        if not all_indiv:
            return []

        all_indiv.sort(key=lambda ind: ind.baseline_fitness, reverse=True)
        return all_indiv[:k]

    # Solver ES: Initialization + One Generation
    def initialize_solvers(
        self,
        mu: int,
        theta_dim: int,
        init_scale: float = 1.0,
    ) -> None:
        """
        Initialize ES Population Of Solvers (Theta Vectors For GreedySolver).
        """
        self.solver_population = SolverPop.initialize_population(
            rng=self.rng,
            mu=mu,
            theta_dim=theta_dim,
            init_scale=init_scale,
        )

    def evolve_solvers_one_generation(
        self,
        k_envs: int = 16,
        mu: int = 5,
        lambd: int = 10,
        sigma: float = 0.5,
    ) -> None:
        """
        Run One ES Generation Over The Solver Population:

            1. Sample k_envs Mazes From Archive.
            2. Evaluate Each Solver On These Mazes.
            3. Apply (mu + lambda)-ES To Update Population.

        Note:
            We Let GreedySolver Internally Decide The Effective Step Count
            For Failed Runs (No External Fixed Penalty Here).
        """
        if not self.solver_population:
            return

        envs = self.sample_envs_for_solver(k_envs)
        if not envs:
            return

        SolverPop.evaluate_on_envs(
            population=self.solver_population,
            envs=envs,
            rng=self.rng,
        )

        self.solver_population = SolverPop.evolve_es(
            rng=self.rng,
            population=self.solver_population,
            mu=mu,
            lambd=lambd,
            sigma=sigma,
        )

    def refresh_solver_fitness(self) -> None:
        """
        Use Current Best Solver In The Population To Evaluate
        Each Maze In The Archive And Write solver_fitness.

        solver_fitness = -steps, Where steps Is Returned By GreedySolver.
        We Do Not Override It With A Fixed Penalty Here.
        """
        if not self.solver_population:
            return

        # Best Solver By ES Fitness
        best = max(self.solver_population, key=lambda s: s.fitness)
        solver = Solver.GreedySolver(theta=best.theta)

        for indiv in self.archive.iter_individuals():
            maze = Maze(indiv.genome)
            success, steps = solver.solve(maze)
            # steps Already Encodes Partial Progress Even If success=False
            indiv.solver_fitness = -float(steps)


    # Specially made for early stage solver training
    def sample_easy_envs_for_solver(self, k: int) -> list[Individual]:
        """
        Sample Easier Mazes For Solver Training:
        - Shorter Baseline Path
        - Fewer Checkpoints
        """
        all_indiv = self.archive.iter_individuals()
        scored = []

        for ind in all_indiv:
            maze = Maze(ind.genome)
            _, steps = maze.evaluate_structure()
            cp = maze.checkpoint_count()
            scored.append((steps, cp, ind))

        scored.sort(key=lambda t: (t[0], t[1]))
        return [ind for (_, _, ind) in scored[:k]]