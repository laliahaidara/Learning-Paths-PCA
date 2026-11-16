"""Synthetic instance generators for the experiments.

Three families from the knapsack literature, from easiest to hardest for
branch and bound: uncorrelated, weakly correlated and strongly correlated
(score = time + constant, so every ratio is similar and bounds prune little).
"""

import random

from algorithms import LearningModule, validate_instance

FAMILIES = ("uncorrelated", "weakly_correlated", "strongly_correlated")


def generate_instance(n, family, budget_ratio, seed, max_time=20):
    """Return ``(modules, budget)`` for a reproducible random instance.

    ``n`` modules with times in 1..max_time; the scores depend on ``family``;
    the budget T is ``budget_ratio`` times the sum of all times (at least 1).
    The same ``seed`` always gives the same instance.
    """
    rng = random.Random(seed)
    times = [rng.randint(1, max_time) for _ in range(n)]
    if family == "uncorrelated":
        scores = [rng.randint(1, max_time) for _ in times]  # score independent of time
    elif family == "weakly_correlated":
        noise = max(1, max_time // 10)
        scores = [max(1, t + rng.randint(-noise, noise)) for t in times]  # score ~ time
    elif family == "strongly_correlated":
        scores = [t + max_time // 2 for t in times]  # score = time + constant
    else:
        raise ValueError(f"unknown family: {family}")
    modules = [LearningModule(f"M{i + 1}", t, s) for i, (t, s) in enumerate(zip(times, scores))]
    budget = max(1, int(budget_ratio * sum(times)))
    validate_instance(modules, budget)
    return modules, budget
