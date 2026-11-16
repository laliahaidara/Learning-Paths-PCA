"""Shared data model for every algorithm.

Problem (0/1 knapsack for learning paths): given modules i = 1..n with a time
cost (positive integer, e.g. hours) and a score (non-negative integer), pick a
subset whose total time is at most the budget T and whose total score is as
large as possible. Each module can be taken at most once.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class LearningModule:
    """One selectable learning module (an "item" of the knapsack)."""

    name: str
    time: int  # cost w_i: how much of the time budget the module uses
    score: int  # value v_i: how much the module is worth


@dataclass
class Result:
    """What every algorithm returns, so the experiments can treat them alike."""

    algorithm: str  # label used in the tables and plots
    value: int  # total score of the chosen modules
    # chosen modules (0-based positions); empty when the algorithm does not reconstruct them
    indices: List[int] = field(default_factory=list)
    operations: int = 0  # machine-independent cost counter (nodes, table cells, comparisons...)


def validate_instance(modules, budget):
    """Raise ValueError unless the instance satisfies the problem's assumptions.

    Budget must be non-negative, every time at least 1 and every score at
    least 0. (A time of 0 would make a module free and break the DP indexing.)
    """
    if budget < 0:
        raise ValueError("negative budget")
    for m in modules:
        if m.time < 1 or m.score < 0:
            raise ValueError(f"invalid module: {m}")


def evaluate(modules, indices):
    """Return (total time, total score) of a set of chosen module indices."""
    total_time = sum(modules[i].time for i in indices)
    total_score = sum(modules[i].score for i in indices)
    return total_time, total_score
