"""Pins the worked example of the report (6 modules, T = 10 h) to the code.

Run from the project root:   python -m unittest discover -s tests -v
If a number here changes, the "Test instance" section of the report must change too.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algorithms import (  # noqa: E402
    LearningModule,
    branch_and_bound,
    dp_memoized,
    dp_table,
    dp_vector,
    validate_instance,
)
from experiment import test_instance  # noqa: E402

# D[i][t] for i = 0..6 modules considered and t = 0..10 hours (the test instance of the report).
EXPECTED_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    [0, 10, 10, 10, 10, 46, 56, 56, 56, 56, 56],
    [0, 10, 10, 10, 10, 46, 56, 56, 56, 56, 91],
    [0, 10, 15, 25, 25, 46, 56, 61, 71, 71, 91],
    [0, 10, 15, 25, 30, 46, 56, 61, 71, 76, 91],
    [0, 10, 16, 25, 31, 46, 56, 62, 71, 77, 91],
]
OPTIMAL_SET = ["Python Basics", "Data Structures"]


class WorkedExample(unittest.TestCase):
    def setUp(self):
        self.modules, self.budget = test_instance()

    def names(self, result):
        return [self.modules[i].name for i in result.indices]

    def test_dp_table_and_reconstruction(self):
        result, table = dp_table(self.modules, self.budget)
        self.assertEqual(table, EXPECTED_TABLE)
        self.assertEqual(result.value, 91)
        self.assertEqual(self.names(result), OPTIMAL_SET)
        self.assertEqual(result.operations, 6 * 11)  # n * (T + 1) cells

    def test_every_exact_algorithm_finds_the_optimum(self):
        for algorithm in (dp_memoized, branch_and_bound):
            result = algorithm(self.modules, self.budget)
            self.assertEqual(result.value, 91, result.algorithm)
            self.assertEqual(self.names(result), OPTIMAL_SET, result.algorithm)
        self.assertEqual(dp_vector(self.modules, self.budget).value, 91)

    def test_memoized_dp_visits_fewer_states_than_the_table(self):
        self.assertEqual(dp_memoized(self.modules, self.budget).operations, 32)

    def test_branch_and_bound_matches_dp_on_a_second_instance(self):
        modules = [LearningModule("short", 1, 2), LearningModule("long", 1000, 1000)]
        self.assertEqual(branch_and_bound(modules, 1000).value, 1000)
        self.assertEqual(dp_table(modules, 1000)[0].value, 1000)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            validate_instance([LearningModule("free", 0, 5)], 10)
        with self.assertRaises(ValueError):
            validate_instance([], -1)


if __name__ == "__main__":
    unittest.main()
