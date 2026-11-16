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
    exhaustive_dfs,
    fptas,
    greedy,
    greedy_with_best_item,
    meet_in_the_middle,
    validate_instance,
)
from experiment import test_instance  # noqa: E402

# D[i][t] for i = 0..6 modules considered and t = 0..10 hours (Table 1 of the report).
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
        for algorithm in (exhaustive_dfs, dp_memoized, branch_and_bound, meet_in_the_middle):
            result = algorithm(self.modules, self.budget)
            self.assertEqual(result.value, 91, result.algorithm)
            self.assertEqual(self.names(result), OPTIMAL_SET, result.algorithm)
        self.assertEqual(dp_vector(self.modules, self.budget).value, 91)

    def test_greedy_is_suboptimal_here(self):
        result = greedy(self.modules, self.budget)
        self.assertEqual(result.value, 77)
        self.assertEqual(
            self.names(result), ["Programming Logic", "Python Basics", "Databases", "Git"]
        )
        self.assertEqual(greedy_with_best_item(self.modules, self.budget).value, 77)
        self.assertAlmostEqual((91 - 77) / 91, 0.154, places=3)  # the 15.4% gap

    def test_fptas_guarantee(self):
        self.assertGreaterEqual(fptas(self.modules, self.budget, 0.1).value, 0.9 * 91)

    def test_adversarial_instance(self):
        modules = [LearningModule("short", 1, 2), LearningModule("long", 1000, 1000)]
        self.assertEqual(greedy(modules, 1000).value, 2)  # 99.8% below the optimum
        self.assertEqual(greedy_with_best_item(modules, 1000).value, 1000)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            validate_instance([LearningModule("free", 0, 5)], 10)
        with self.assertRaises(ValueError):
            validate_instance([], -1)


if __name__ == "__main__":
    unittest.main()
