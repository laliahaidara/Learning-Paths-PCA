"""Main experiments for the 0/1 knapsack learning-path project.

Two algorithms are studied: dynamic programming (DP, in three variants) and
branch and bound (B&B). Blocks (choose with --only): demo, correctness, scale,
pseudo, families. Each block prints a summary and writes CSV files and PNG
plots to results/.

    python experiment.py                            # every block
    python experiment.py --quick                    # smaller sizes, faster
    python experiment.py --only demo correctness    # just these blocks
"""

import argparse
import os
import random
import statistics
import time

import matplotlib

matplotlib.use("Agg")  # no window: plots are only saved to files
import matplotlib.pyplot as plt  # noqa: E402

from algorithms import (  # noqa: E402
    LearningModule,
    evaluate,
    validate_instance,
    dp_table,
    dp_vector,
    dp_memoized,
    branch_and_bound,
)
from common import OUTPUT_DIR, measure, run_dp, save_csv  # noqa: E402
from instances import FAMILIES, generate_instance  # noqa: E402


# Every algorithm that must return the true optimum; the experiments loop over this
# dictionary, so adding an exact algorithm here includes it in all the checks.
EXACT = {
    "DP": run_dp,
    "DP-vector": dp_vector,
    "DP-memo": dp_memoized,
    "B&B": branch_and_bound,
}


def save_figure(name):
    """Save the current matplotlib figure to results/<name> and close it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150)
    plt.close()


def brute_force_value(modules, budget):
    """Reference optimum by listing every feasible subset (only for n <= 16).

    This is a checking helper for the correctness block, not one of the compared
    algorithms: it keeps the (time, score) pair of every subset that fits.
    """
    subsets = [(0, 0)]
    for m in modules:
        subsets += [(t + m.time, s + m.score) for t, s in subsets if t + m.time <= budget]
    return max(s for _, s in subsets)


def test_instance():
    """The worked example of the report: 6 modules and a budget of T = 10 hours.

    The best choice is "Python Basics" + "Data Structures" (10 hours, 91 points).
    """
    modules = [
        LearningModule("Programming Logic", 1, 10),
        LearningModule("Python Basics", 5, 46),
        LearningModule("Data Structures", 5, 45),
        LearningModule("Databases", 2, 15),
        LearningModule("Networking", 3, 20),
        LearningModule("Git", 1, 6),
    ]
    validate_instance(modules, 10)
    return modules, 10


def exp_demo():
    """Block ``demo``: solve the test instance with every algorithm.

    Prints the instance, the DP table and each algorithm's value / time used /
    operations / chosen modules.
    """
    modules, T = test_instance()  # same numbers as the worked example in the report
    print("\n=== TEST INSTANCE (T = %d h) ===" % T)
    for i, m in enumerate(modules, 1):
        print(f"{i}. {m.name:<20} time={m.time}  score={m.score}  ratio={m.score / m.time:.1f}")
    _, D = dp_table(modules, T)  # D[i][t] = best score with the first i modules and t hours
    print("\nDP table (rows i = modules considered, columns t = 0..T):")
    print("i\\t " + " ".join(f"{t:>3}" for t in range(T + 1)))
    for i, row in enumerate(D):
        print(f"{i:>3} " + " ".join(f"{x:>3}" for x in row))
    print("\nResults:")
    # The DP vector only returns the value (no path), hence the "no reconstruction" label below.
    for algorithm in EXACT.values():
        r = algorithm(modules, T)
        names = [modules[i].name for i in r.indices] if r.indices else "(no reconstruction)"
        used = evaluate(modules, r.indices)[0] if r.indices else "-"
        print(
            f"{r.algorithm:<15} value={r.value:<4} time used={used!s:<3} "
            f"ops={r.operations:<5} {names}"
        )


def edge_cases():
    """Edge cases that every exact algorithm must solve identically.

    Empty list, zero budget, nothing fits, everything fits, and ties.
    """
    cases = [  # (modules, budget) pairs chosen to break naive implementations
        ([], 5),
        ([LearningModule("a", 3, 5)], 0),
        ([LearningModule("a", 6, 9), LearningModule("b", 7, 9)], 5),
        ([LearningModule("a", 1, 2), LearningModule("b", 1, 3)], 100),
        ([LearningModule("a", 2, 4), LearningModule("b", 2, 4)], 2),
    ]
    for modules, T in cases:
        values = {name: algorithm(modules, T).value for name, algorithm in EXACT.items()}
        assert len(set(values.values())) == 1, f"divergence in an edge case: {values}"
    print(f"Edge cases: {len(cases)} OK")


def exp_correctness(count, seed=2024):
    """Block ``correctness``: compare every algorithm with a brute-force reference.

    Runs ``count`` random instances (n <= 16, all families and budgets) and checks
    that each chosen set is feasible and has the reported score.
    """
    edge_cases()
    rng = random.Random(seed)  # fixed seed: a failure can always be reproduced
    failures = []
    for k in range(count):
        n = rng.randint(0, 16)
        modules, T = generate_instance(
            n,
            rng.choice(FAMILIES),
            rng.choice([0.1, 0.3, 0.5, 0.7, 1.0]),
            rng.randrange(10**9),
            rng.choice([5, 20]),
        )
        reference = brute_force_value(modules, T)  # the oracle (n <= 16)
        for name, algorithm in EXACT.items():
            r = algorithm(modules, T)
            if r.value != reference:
                failures.append((k, name, "value", r.value, reference))
            # Re-evaluate the chosen set from scratch: it must fit and match the reported score.
            # (The DP vector keeps no path, so it is skipped.)
            if r.indices or (r.value == 0 and name != "DP-vector"):
                used, score = evaluate(modules, r.indices)
                if used > T or score != r.value:
                    failures.append((k, name, "infeasible/inconsistent solution", used, score))
    print(f"Correctness: {count} random instances (n <= 16), failures = {len(failures)}")
    for f in failures[:5]:
        print("  FAILURE:", f)
    assert not failures, "there are divergences: investigate before moving on"


def exp_scale(sizes):
    """Block ``scale``: DP for large n; confirms that the cost is exactly n * (T + 1)."""
    rows = []
    for n in sizes:
        modules, T = generate_instance(n, "uncorrelated", 0.5, 7 * n)
        for name, algorithm in {"DP": run_dp, "DP-vector": dp_vector}.items():
            t, r = measure(algorithm, modules, T, repetitions=3)
            rows.append(
                {
                    "n": n,
                    "T": T,
                    "algorithm": name,
                    "time_s": t,
                    "operations": r.operations,
                    "n_times_T": n * (T + 1),
                }
            )
        print(f"  n={n} T={T} ok")
    save_csv("scale.csv", rows)

    _, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("DP", "DP-vector"):
        points = [(r["n"], r["time_s"], r["operations"]) for r in rows if r["algorithm"] == name]
        axes[0].plot([p[0] for p in points], [p[1] for p in points], marker="o", label=name)
        axes[1].plot([p[0] for p in points], [p[2] for p in points], marker="o", label=name)
    axes[1].plot(
        [r["n"] for r in rows if r["algorithm"] == "DP"],
        [r["n_times_T"] for r in rows if r["algorithm"] == "DP"],
        "k--",
        label="n·(T+1)",
    )
    for ax, label in zip(axes, ("time (s)", "operations")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (modules)")
        ax.set_ylabel(label + " (log)")
        ax.legend()
        ax.grid(alpha=0.3)
    save_figure("fig_scale.png")


def exp_pseudopolynomial(n, factors):
    """Block ``pseudo``: same modules with all times (and T) multiplied by a factor; DP cost grows
    with T, showing it is pseudo-polynomial, while B&B hardly notices.
    """
    base, T0 = generate_instance(n, "uncorrelated", 0.5, 99)
    rows = []
    for factor in factors:
        # Scaling every time and T by the same factor keeps the optimum identical but makes
        # the table T times wider: only the DP feels it.
        modules = [LearningModule(m.name, m.time * factor, m.score) for m in base]
        T = T0 * factor
        for name, algorithm in {"B&B": branch_and_bound, "DP": run_dp}.items():
            t, r = measure(algorithm, modules, T, repetitions=3)
            rows.append(
                {
                    "factor": factor,
                    "T": T,
                    "algorithm": name,
                    "time_s": t,
                    "operations": r.operations,
                }
            )
        print(f"  factor x{factor} ok")
    save_csv("pseudopolynomial.csv", rows)

    plt.figure(figsize=(7, 4))
    for name in ("B&B", "DP"):
        points = [(r["T"], r["operations"]) for r in rows if r["algorithm"] == name]
        plt.plot(*zip(*points), marker="o", label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("T (budget, in the chosen time unit)")
    plt.ylabel("operations (log)")
    plt.title(f"Pseudo-polynomial behavior (n={n} fixed)")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure("fig_pseudopolynomial.png")


def exp_families(ns, count, max_time=1000):
    """Block ``families``: operations of B&B and DP by instance family (strongly correlated
    instances hurt B&B the most).
    """
    rows = []
    for family in ("uncorrelated", "strongly_correlated"):
        for n in ns:
            ops_bb, ops_dp = [], []
            for s in range(count):
                modules, T = generate_instance(n, family, 0.5, 17 * s + n, max_time)
                ops_bb.append(branch_and_bound(modules, T).operations)
                ops_dp.append(dp_vector(modules, T).operations)
            rows.append(
                {
                    "family": family,
                    "n": n,
                    "bb_mean": statistics.mean(ops_bb),
                    "bb_max": max(ops_bb),
                    "dp_mean": statistics.mean(ops_dp),
                }
            )
            r = rows[-1]
            print(
                f"  {family:<20} n={n:<3} B&B mean={r['bb_mean']:>11.0f} max={r['bb_max']:>11}"
                f"  DP={r['dp_mean']:>9.0f}"
            )
    save_csv("exact_by_family.csv", rows)

    plt.figure(figsize=(7, 4))
    for family, style in (("uncorrelated", "-"), ("strongly_correlated", "--")):
        cells = [r for r in rows if r["family"] == family]
        plt.plot(
            [r["n"] for r in cells],
            [r["bb_mean"] for r in cells],
            style + "o",
            label=f"B&B ({family})",
        )
    cells = [r for r in rows if r["family"] == "uncorrelated"]
    plt.plot([r["n"] for r in cells], [r["dp_mean"] for r in cells], "k-s", label="DP (any family)")
    plt.yscale("log")
    plt.xlabel("n (modules)")
    plt.ylabel("operations (log)")
    plt.title(f"B&B and DP by family (max time per module = {max_time})")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    save_figure("fig_exact_by_family.png")


def main():
    """Parse the command line and run the selected blocks."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="reduced version of the experiments")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="blocks: demo correctness scale pseudo families",
    )
    args = parser.parse_args()
    q = args.quick  # --quick shrinks sizes and repetitions (a few seconds instead of a minute)
    blocks = {
        "demo": exp_demo,
        "correctness": lambda: exp_correctness(300 if q else 1200),
        "scale": lambda: exp_scale([50, 100, 200] if q else [50, 100, 200, 400, 600]),
        "pseudo": lambda: exp_pseudopolynomial(
            16 if q else 18, [1, 4, 16, 64] if q else [1, 4, 16, 64, 256, 1024]
        ),
        "families": lambda: exp_families([20, 30] if q else [20, 30, 40], 3 if q else 5),
    }
    for name in args.only or blocks:
        print(f"\n### block: {name}")
        start = time.perf_counter()
        blocks[name]()
        print(f"   ({time.perf_counter() - start:.1f} s)")


if __name__ == "__main__":
    main()
