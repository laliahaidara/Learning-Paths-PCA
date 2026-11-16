"""Helpers shared by every experiment script: CSV output and timing.

Kept in one place so ``experiment.py``, ``experiment_validation.py``,
``experiment_real.py`` and ``experiment_grasp.py`` do not repeat them.
"""

import csv
import os
import statistics
import time

from algorithms import dp_table, fptas

OUTPUT_DIR = "results"  # every CSV and PNG produced by the experiments goes here


def save_csv(name, rows):
    """Write a list of dict rows to results/<name> (header = keys of the first row)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, name), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def measure(function, *args, repetitions=3):
    """Run ``function(*args)`` ``repetitions`` times; return (median time in s, last result).

    The median (not the mean) keeps one slow run, caused by the operating
    system, from distorting the wall-clock figures.
    """
    times, result = [], None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function(*args)
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def run_dp(modules, budget):
    """Run the table DP and keep only its Result (the DP matrix is dropped)."""
    return dp_table(modules, budget)[0]


def fptas_10(modules, budget):
    """FPTAS with epsilon = 0.1, the setting used in the report."""
    return fptas(modules, budget, 0.1)
