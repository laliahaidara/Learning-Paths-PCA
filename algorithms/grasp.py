"""GRASP (Greedy Randomized Adaptive Search Procedure) metaheuristic.

Each iteration has two phases:
  1. Greedy randomized construction: instead of always taking the best
     remaining ratio (pure greedy), draw uniformly from a Restricted
     Candidate List (RCL) holding the top-alpha fraction of the modules that
     still fit. alpha -> 0 approaches the pure greedy; alpha = 1 is random.
  2. Local search (first improvement): add any module that fits, and swap a
     chosen module for an outside one with a higher score when it fits. Each
     accepted move strictly raises the total score, so the search terminates.
The best solution over all iterations is returned. There is no optimality
guarantee: it is a heuristic, measured empirically against the exact DP.
"""

import random

from .model import Result
from .sorting import sort_by_ratio


def _construct(order, modules, budget, alpha, rng, counter):
    """Build one solution; returns (chosen indices, time used)."""
    chosen, used = [], 0
    in_solution = [False] * len(modules)
    candidates = list(order)  # modules in decreasing score/time order
    while True:
        fitting = []
        for i in candidates:
            counter[0] += 1
            if not in_solution[i] and used + modules[i].time <= budget:
                fitting.append(i)
        if not fitting:
            return chosen, used
        # RCL = first ceil(alpha * |fitting|) candidates (the best ratios), at least 1.
        # Integer arithmetic avoids floating-point surprises in the ceiling.
        rcl_size = max(1, -(-int(alpha * 100 * len(fitting)) // 100))
        pick = fitting[rng.randrange(rcl_size)]
        chosen.append(pick)
        in_solution[pick] = True
        used += modules[pick].time


def _local_search(chosen, used, modules, budget, counter):
    """Improve a solution with add and swap moves; returns (chosen, used, value)."""
    value = sum(modules[i].score for i in chosen)
    in_solution = [False] * len(modules)
    for i in chosen:
        in_solution[i] = True
    improved = True
    while improved:
        improved = False
        # Add move: any outside module that still fits.
        for i in range(len(modules)):
            counter[0] += 1
            if not in_solution[i] and used + modules[i].time <= budget:
                chosen.append(i)
                in_solution[i] = True
                used += modules[i].time
                value += modules[i].score
                improved = True
        # Swap move: exchange a chosen module for an outside one with a higher score.
        for out in range(len(modules)):
            if in_solution[out]:
                continue
            for inside in list(chosen):
                counter[0] += 1
                if (
                    modules[out].score > modules[inside].score
                    and used - modules[inside].time + modules[out].time <= budget
                ):
                    chosen.remove(inside)
                    in_solution[inside] = False
                    chosen.append(out)
                    in_solution[out] = True
                    used += modules[out].time - modules[inside].time
                    value += modules[out].score - modules[inside].score
                    improved = True
                    break
    return chosen, used, value


def grasp(modules, budget, iterations=20, alpha=None, seed=0):
    """Run GRASP for ``iterations`` rounds and return the best solution found.

    ``alpha=None`` draws a different alpha in [0.15, 0.9] in each iteration;
    the fixed ``seed`` makes every run reproducible.
    """
    rng = random.Random(seed)
    counter = [0]
    order = sort_by_ratio(modules, counter)
    best_value, best_set = 0, []
    for _ in range(iterations):
        step_alpha = rng.uniform(0.15, 0.9) if alpha is None else alpha
        chosen, used = _construct(order, modules, budget, step_alpha, rng, counter)
        chosen, used, value = _local_search(chosen, used, modules, budget, counter)
        if value > best_value:
            best_value, best_set = value, sorted(chosen)
    return Result("GRASP", best_value, best_set, counter[0])
