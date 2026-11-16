"""Algorithms for the 0/1 knapsack applied to learning-path selection.

One file per algorithm: dynamic programming (``dp.py``) and branch and bound
(``branch_and_bound.py``). This package re-exports the public functions so the
experiments can write ``from algorithms import dp_table, branch_and_bound``.
"""

from .model import LearningModule, Result, validate_instance, evaluate
from .dp import dp_table, dp_vector, dp_memoized
from .branch_and_bound import branch_and_bound

__all__ = [
    "LearningModule",
    "Result",
    "validate_instance",
    "evaluate",
    "dp_table",
    "dp_vector",
    "dp_memoized",
    "branch_and_bound",
]
