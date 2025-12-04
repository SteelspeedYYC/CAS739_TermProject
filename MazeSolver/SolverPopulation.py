# Build the solvers, pick the best one and evo
from dataclasses import dataclass
import numpy as np
from Maze.Maze import Maze
from MazeSolver.Solver import AStarSolver, GreedySolver
from Archive import Individual


@dataclass
class SolverIndividual:
    """
    One Candidate Solver In The ES Population.

    theta: Parameter Vector For AStarSolver's Heuristic
    fitness: -Average Steps Over A Set Of Mazes
    """
    theta: np.ndarray
    fitness: float = 0.0


def initialize_population(
    rng: np.random.Generator,
    mu: int,
    theta_dim: int,
    init_scale: float = 1.0,
) -> list[SolverIndividual]:
    """
    Create An Initial Population Of mu SolverIndividuals
    With Random Gaussian Parameters.
    """
    population: list[SolverIndividual] = []
    for _ in range(mu):
        theta = rng.normal(loc=0.0, scale=init_scale, size=theta_dim)
        population.append(SolverIndividual(theta=theta, fitness=0.0))
    return population


def evaluate_on_envs(
    population: list[SolverIndividual],
    envs: list[Individual],
    rng: np.random.Generator,
) -> None:
    """
    Evaluate Each SolverIndividual On A List Of Maze Environments.

    For Each Solver:
        - Compute Baseline Steps With Maze.evaluate_structure()
        - Run GreedySolver(theta) To Get solver_steps
        - Define A Relative Score:

              score_env = min(1.0, baseline_steps / solver_steps)

          (Higher Is Better, 1.0 Means Matching Or Beating Baseline)

        - fitness = Average score_env Over All Envs.
    """
    if not envs:
        for s in population:
            s.fitness = 0.0
        return

    for s in population:
        solver = GreedySolver(theta=s.theta)
        total_score = 0.0
        count = 0

        for env in envs:
            maze = Maze(env.genome)

            # Baseline evaluation: returns (fitness, steps)
            _, baseline_steps = maze.evaluate_structure()

            # baseline_steps <= 0, skip
            if baseline_steps <= 0:
                continue

            success, solver_steps = solver.solve(maze)
            solver_steps = float(solver_steps)

            if solver_steps <= 0:
                continue

            # Relative performance: smaller solver_steps => larger ratio
            ratio = baseline_steps / solver_steps

            # Cap At 1.0: Solver Can't Exceed "Best Possible"
            score_env = min(1.0, ratio)

            total_score += score_env
            count += 1

        if count == 0:
            s.fitness = 0.0
        else:
            s.fitness = total_score / count  # In [0, 1]


def evolve_es(
    rng: np.random.Generator,
    population: list[SolverIndividual],
    mu: int,
    lambd: int,
    sigma: float = 0.2,
) -> list[SolverIndividual]:
    """
    Elitist (mu + lambda)-ES On The Solver Population.

        1. Sort Current Population By Fitness (Descending).
        2. Keep The Best mu Individuals As Parents (Elitism).
        3. Generate lambd Offspring By Adding Gaussian Noise To Parents.
        4. New Population = Parents + Offspring.

    Note:
        - We Do NOT Modify Parent Thetas In-Place.
        - sigma Should Be Relatively Small (e.g. 0.1 ~ 0.3) So That
          Good Solvers Are Not Immediately Destroyed.
    """
    if not population:
        return population

    # 1. Sort By Fitness (Best First)
    pop_sorted = sorted(population, key=lambda s: s.fitness, reverse=True)

    # 2. Elitism: Keep Top mu As Parents
    parents = pop_sorted[:mu]

    # 3. Generate Offspring
    offspring: list[SolverIndividual] = []
    for _ in range(lambd):
        # Randomly Pick One Parent To Mutate
        p = rng.choice(parents)

        # Copy Theta (Do NOT Modify Parent In-Place)
        child_theta = p.theta.copy()

        # Add Gaussian Noise
        noise = rng.normal(loc=0.0, scale=sigma, size=child_theta.shape)
        child_theta += noise

        offspring.append(SolverIndividual(theta=child_theta, fitness=0.0))

    # 4. New Population = Parents (Elites) + Children
    return parents + offspring