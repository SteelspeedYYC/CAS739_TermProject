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
    alpha_cp: float = 0.5,
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

    alpha_cp = float(alpha_cp)
    alpha_cp = max(0.0, min(1.0, alpha_cp))

    for s in population:
        solver = GreedySolver(theta=s.theta)
        total_score = 0.0
        count = 0

        for env in envs:
            maze = Maze(env.genome)

            # 1) baseline
            _, baseline_steps = maze.evaluate_structure_noCP()
            
            # 2) solver run
            success, raw_steps, cp_ratio = solver.solve_with_stats(maze)
            
            # A. Completion Points
            completion_score = 1.0 if success else 0.0
            
            # B. CP Points
            task_score = 2.0 * cp_ratio
            
            # C. Eff Points
            max_tolerable_steps = maze.height * maze.width * 2
            efficiency_score = max(0.0, 1.0 - (raw_steps / max_tolerable_steps))
            # MAX = 1.0 + 2.0 + 0.5 = 3.5
            score_env = completion_score + task_score + (0.5 * efficiency_score)
            
            # Normal 0~1
            rel_score = score_env / 3.5
            

            total_score += rel_score
            count += 1

        avg_score = total_score / max(count, 1)
        s.fitness = avg_score


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