# runner.py for whole CoEV process

import numpy as np
from Maze.Maze import Maze
from Maze.IGenerator import IGenerator
from Maze.Generator import PerlinMazeGenerator, DFSMazeGenerator
from Archive import MapElitesArchive, Individual
from Controller import ExperimentController
import MazeSolver.SolverPopulation as SolverPop
from MazeSolver.Solver import GreedySolver


# Helper

def pick_solvable_envs(archive: MapElitesArchive,
                       rng: np.random.Generator,
                       k: int) -> list[Individual]:
    """
    From archive picking baseline_fitness > -500 (solvable) maze
    Max do k picking.
    """
    all_indiv = archive.iter_individuals()
    solvable = [ind for ind in all_indiv if ind.baseline_fitness > -500.0]

    if not solvable:
        return []

    if len(solvable) <= k:
        return solvable

    idx = rng.choice(len(solvable), size=k, replace=False)
    return [solvable[int(i)] for i in idx]


def update_solver_fitness_for_archive(ctrl: ExperimentController,
                                      best_theta: np.ndarray,
                                      alpha_cp: float = 0.5) -> None:
    """
    Using current best solver for archive's mazes to calculate solver_fitness
    Using same eval as SolverPopulation.evaluate_on_envs()
    """
    solver = GreedySolver(theta=best_theta)
    for indiv in ctrl.archive.iter_individuals():
        maze = Maze(indiv.genome)

        # baseline_noCP
        _, baseline_steps = maze.evaluate_structure_noCP()
        baseline_steps = float(max(1.0, baseline_steps))

        success, raw_steps, cp_ratio = solver.solve_with_stats(maze)
        raw_steps = float(max(1.0, raw_steps))
        cp_ratio = float(np.clip(cp_ratio, 0.0, 1.0))

        # Failure condition
        if not success:
            max_steps = int(solver.max_steps_factor * maze.height * maze.width)
            raw_steps = float(max_steps)
            cp_factor = (1.0 - alpha_cp)
        else:
            cp_factor = (1.0 - alpha_cp) + alpha_cp * cp_ratio

        speed_ratio = baseline_steps / raw_steps
        rel_score = speed_ratio * cp_factor
        rel_score = float(np.clip(rel_score, 0.0, 1.0))

        indiv.solver_fitness = rel_score


def snapshot_archive(archive: MapElitesArchive, gen: int) -> dict:
    """
    Quick save of archive
    """
    cells_data: list[dict] = []

    for r, row in enumerate(archive.grid):
        for c, indiv in enumerate(row):
            if indiv is None:
                continue
            cells_data.append(
                {
                    "row": r,
                    "col": c,
                    "baseline_fitness": float(indiv.baseline_fitness),
                    "solver_fitness": (
                        float(indiv.solver_fitness)
                        if indiv.solver_fitness is not None
                        else None
                    ),
                    "features": tuple(float(x) for x in indiv.features),
                    # genome to list
                    "genome": indiv.genome.tolist(),
                }
            )

    return {
        "gen": gen,
        "cells": cells_data,
    }


def eval_solver_vs_baseline(theta: np.ndarray,
                            envs: list[Individual],
                            alpha_cp: float = 0.5) -> tuple[float, float, float]:
    """
    eval solver on centain given mazes
      - return (avg_solver_steps, avg_rel_score, avg_baseline_steps)
    """
    if not envs:
        return 0.0, 0.0, 0.0

    solver = GreedySolver(theta=theta)
    total_steps = 0.0
    total_rel = 0.0
    total_base = 0.0
    count = 0

    for indiv in envs:
        maze = Maze(indiv.genome)

        # baseline_noCP
        _, baseline_steps = maze.evaluate_structure_noCP()
        baseline_steps = float(max(1.0, baseline_steps))

        success, raw_steps, cp_ratio = solver.solve_with_stats(maze)
        raw_steps = float(max(1.0, raw_steps))
        cp_ratio = float(np.clip(cp_ratio, 0.0, 1.0))

        # Same logic as evaluate_on_envs
        if not success:
            max_steps = int(solver.max_steps_factor * maze.height * maze.width)
            raw_steps = float(max_steps)
            cp_factor = (1.0 - alpha_cp)
        else:
            cp_factor = (1.0 - alpha_cp) + alpha_cp * cp_ratio

        speed_ratio = baseline_steps / raw_steps
        rel_score = speed_ratio * cp_factor
        rel_score = float(np.clip(rel_score, 0.0, 1.0))

        total_steps += raw_steps
        total_rel += rel_score
        total_base += baseline_steps
        count += 1

    avg_steps = total_steps / count
    avg_rel = total_rel / count
    avg_base = total_base / count
    return avg_steps, avg_rel, avg_base


# Main CoEV

def run_coevolution_experiment(
    seed: int = 123,
    n_init_mazes: int = 200,
    maze_size: int = 16,
    bins_f1: int = 10,
    bins_f2: int = 10,
    alpha_cp: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    """
    Main CoEV loop controlling
      - Return (solver_logs, maze_logs) for plotting
    """
    rng = np.random.default_rng(seed)

    # 1) Init and construct archive / controller
    pn_gen = PerlinMazeGenerator(size=maze_size)
    dfs_gen = DFSMazeGenerator(size=maze_size)

    archive = MapElitesArchive(bins_f1=bins_f1, bins_f2=bins_f2)
    archive_snapshots: list[dict] = []

    ctrl = ExperimentController(generator=pn_gen, archive=archive, rng=rng)

    # 2) Init archive: mix Perlin + DFS, half half with total num in n_init_mazes
    ctrl.initialize_archive_mixed(
        n=n_init_mazes,
        perlin_gen=pn_gen,
        dfs_gen=dfs_gen,
    )

    print(f"[Init] Archive initialized with up to {n_init_mazes} mazes.")

    # 3) Pick 100 solvable mazes for Phase 1
    train_envs_phase1 = pick_solvable_envs(archive, rng, k=100)
    print(f"[Phase1] Picked {len(train_envs_phase1)} solvable mazes for initial solver training.")

    # 4) init solver population
    mu = 10
    theta_dim = 4
    ctrl.initialize_solvers(mu=mu, theta_dim=theta_dim, init_scale=1.0)

    # 5) Phase 1: 51 pre-training
    lambd = 20
    sigma = 0.3
    p1_trains = 51

    for gen in range(p1_trains):
        # Use SolverPopulation for fixed envs
        SolverPop.evaluate_on_envs(
            population=ctrl.solver_population,
            envs=train_envs_phase1,
            rng=rng,
            alpha_cp=alpha_cp,
        )
        ctrl.solver_population = SolverPop.evolve_es(
            rng=rng,
            population=ctrl.solver_population,
            mu=mu,
            lambd=lambd,
            sigma=sigma,
        )

        # See performance and convergnce
        fitnesses = [s.fitness for s in ctrl.solver_population]
        best_fit = max(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)
        if gen % 5 == 0:
            print(f"[Phase1 Gen {gen}] solver best={best_fit:.3f}, avg={avg_fit:.3f}")

    # 6) Post phase, send solver score to map
    best_solver = max(ctrl.solver_population, key=lambda s: s.fitness)
    update_solver_fitness_for_archive(ctrl, best_solver.theta, alpha_cp=alpha_cp)
    print("[Phase1] Solver fitness written to archive.")
    archive_snapshots.append(snapshot_archive(archive, gen=-1))

    # CoEV loop

    n_coev_gens = 20

    solver_logs: list[dict] = []
    maze_logs: list[dict] = []
    final_best_theta = None
    final_viz_envs: list[Individual] = []

    for gen in range(n_coev_gens):
        print(f"\n[CoEV Gen {gen}]")

        # 1) Maze turn
        ctrl.step_evolution(
            offspring_per_step=10,
            use_solver_fitness=True,
        )

        # 2) Solver turn
        cur_train_envs = pick_solvable_envs(archive, rng, k=50)
        if cur_train_envs:
            SolverPop.evaluate_on_envs(
                population=ctrl.solver_population,
                envs=cur_train_envs,
                rng=rng,
                alpha_cp=alpha_cp,
            )
            ctrl.solver_population = SolverPop.evolve_es(
                rng=rng,
                population=ctrl.solver_population,
                mu=mu,
                lambd=lambd,
                sigma=sigma,
            )

        # 3) Updating archive solver_fitness
        best_solver = max(ctrl.solver_population, key=lambda s: s.fitness)
        update_solver_fitness_for_archive(ctrl, best_solver.theta, alpha_cp=alpha_cp)

        # 4) Checking (eval) solver on random solvable 20 mazes
        eval_envs = pick_solvable_envs(archive, rng, k=20)
        avg_steps, avg_rel, avg_base = eval_solver_vs_baseline(
            best_solver.theta,
            eval_envs,
            alpha_cp=alpha_cp,
        )
        solver_logs.append(
            {
                "gen": gen,
                "avg_solver_steps": avg_steps,
                "avg_rel": avg_rel,
                "avg_baseline_steps": avg_base,
            }
        )
        print(
            f"  Solver: avg_steps={avg_steps:.1f}, "
            f"avg_rel={avg_rel:.3f}, "
            f"avg_base_steps={avg_base:.1f}"
        )

        # 5) Checking maze with mean of baseline_fitness mean of solver_fitness
        all_indiv = archive.iter_individuals()
        base_vals = [ind.baseline_fitness for ind in all_indiv]
        all_indiv = archive.iter_individuals()
        solver_vals = [
            ind.solver_fitness
            for ind in all_indiv
            if ind.solver_fitness is not None
        ]

        avg_base_fit = float(sum(base_vals) / len(base_vals)) if base_vals else 0.0
        avg_solver_fit = (
            float(sum(solver_vals) / len(solver_vals)) if solver_vals else 0.0
        )

        maze_logs.append(
            {
                "gen": gen,
                "avg_baseline_fitness": avg_base_fit,
                "avg_solver_fitness": avg_solver_fit,
            }
        )
        print(
            f"  Maze: avg_baseline_fit={avg_base_fit:.1f}, "
            f"avg_solver_fit={avg_solver_fit:.3f}"
        )
        archive_snapshots.append(snapshot_archive(archive, gen))

        if gen == n_coev_gens - 1:
            final_best_theta = best_solver.theta.copy()
            final_viz_envs = pick_solvable_envs(archive, rng, k=5)

    return solver_logs, maze_logs, final_best_theta, final_viz_envs, archive_snapshots



if __name__ == "__main__":
    solver_logs, maze_logs, final_best_theta, final_viz_envs, archive_snapshots = run_coevolution_experiment()
    print("\n[Done] CoEV finished. You can now load solver_logs / maze_logs in notebook to plot.")
