"""Main experiments for the 0/1 knapsack learning-path project.

Blocks (choose with --only): demo, correctness, time, scale, quality, pseudo,
families, fptas, graph. Each block prints a summary and writes CSV files and
PNG plots to results/.

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
    exhaustive_dfs,
    dfs_paths,
    complete_dag,
    dp_table,
    dp_vector,
    dp_memoized,
    greedy,
    greedy_with_best_item,
    branch_and_bound,
    meet_in_the_middle,
    fptas,
)
from common import OUTPUT_DIR, fptas_10, measure, run_dp, save_csv  # noqa: E402
from instances import FAMILIES, generate_instance  # noqa: E402


# Every algorithm that must return the true optimum; the experiments loop over this
# dictionary, so adding an exact algorithm here includes it in all the checks.
EXACT = {
    "DFS": exhaustive_dfs,
    "DP": run_dp,
    "DP-vector": dp_vector,
    "DP-memo": dp_memoized,
    "B&B": branch_and_bound,
    "MITM": meet_in_the_middle,
}


def save_figure(name):
    """Save the current matplotlib figure to results/<name> and close it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150)
    plt.close()


def gap(optimum, value):
    """Relative distance to the optimum: (OPT - value) / OPT (0 when OPT = 0)."""
    return (optimum - value) / optimum if optimum else 0.0


def test_instance():
    """The worked example of the report: 6 modules and a budget of T = 10 hours.

    The score/time ratios decrease from "Programming Logic" (10.0) to "Git" (6.0),
    so the greedy takes the cheap modules first and ends at 77 points, while the
    optimum (91 points) skips them and takes "Python Basics" + "Data Structures".
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

    Prints the instance, the DP table, each algorithm's value / time used /
    operations / chosen modules, the greedy gap, and the adversarial instance
    where plain greedy is almost 100% off the optimum.
    """
    modules, T = test_instance()  # same numbers as the worked example in the report
    print("\n=== TEST INSTANCE (T = %d h) ===" % T)
    for i, m in enumerate(modules, 1):
        print(f"{i}. {m.name:<20} time={m.time}  score={m.score}  ratio={m.score / m.time:.1f}")
    dp_result, D = dp_table(modules, T)  # D[i][t] = best score with the first i modules and t hours
    print("\nDP table (rows i = modules considered, columns t = 0..T):")
    print("i\\t " + " ".join(f"{t:>3}" for t in range(T + 1)))
    for i, row in enumerate(D):
        print(f"{i:>3} " + " ".join(f"{x:>3}" for x in row))
    print("\nResults:")
    # The DP vector only returns the value (no path), hence the "no reconstruction" label below.
    algorithms = list(EXACT.values()) + [greedy, greedy_with_best_item, fptas_10]
    for algorithm in algorithms:
        r = algorithm(modules, T)
        names = [modules[i].name for i in r.indices] if r.indices else "(no reconstruction)"
        used = evaluate(modules, r.indices)[0] if r.indices else "-"
        print(
            f"{r.algorithm:<15} value={r.value:<4} time used={used!s:<3} "
            f"ops={r.operations:<5} {names}"
        )
    optimum, g = dp_result.value, greedy(modules, T).value
    print(f"\nGreedy gap on this instance: {gap(optimum, g):.1%} ({g} versus {optimum})")

    # Greedy ranks by ratio, takes the tiny module first and then cannot fit the big one.
    adversarial = [LearningModule("short", 1, 2), LearningModule("long", 1000, 1000)]
    print(
        "\nAdversarial instance (T = 1000): short = 1h/2 pts (ratio 2.0), "
        "long = 1000h/1000 pts (ratio 1.0)"
    )
    for algorithm in (greedy, greedy_with_best_item, run_dp):
        r = algorithm(adversarial, 1000)
        print(f"  {r.algorithm:<12} value={r.value}  gap={gap(1000, r.value):.1%}")


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
    """Block ``correctness``: compare every algorithm with the exhaustive DFS.

    Runs ``count`` random instances (n <= 16, all families and budgets), checks
    that each chosen set is feasible and has the reported score, and checks the
    greedy (<= OPT, >= OPT/2) and FPTAS (>= 0.9 OPT) guarantees.
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
        reference = exhaustive_dfs(modules, T).value  # brute force is the oracle (n <= 16)
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
        # Guarantees: greedy <= OPT; greedy+item >= OPT/2; FPTAS(0.1) within 10% of OPT.
        g, gi, f = (
            greedy(modules, T).value,
            greedy_with_best_item(modules, T).value,
            fptas_10(modules, T).value,
        )
        if (
            g > reference
            or gi > reference
            or 2 * gi < reference
            or f > reference
            or f < 0.9 * reference
        ):
            failures.append((k, "greedy/FPTAS", "guarantee violated", (g, gi, f), reference))
    print(f"Correctness: {count} random instances (n <= 16), failures = {len(failures)}")
    for f in failures[:5]:
        print("  FAILURE:", f)
    assert not failures, "there are divergences: investigate before moving on"


def exp_exponential_time(sizes, average_instances=5):
    """Block ``time``: DFS vs B&B vs meet-in-the-middle vs DP as n grows.

    Worst case for DFS (T = 100% of the total time) and an average case (T = 50%).
    Records operations and wall-clock time; plots operations against 2^(n+1).
    """
    rows = []
    for n in sizes:
        # Worst case for the DFS: everything fits, so nothing is pruned (a single instance is
        # enough because DFS cost does not depend on the data). The average case uses several seeds.
        scenarios = {
            "worst case (T=100%)": [generate_instance(n, "uncorrelated", 1.0, n)],
            "average case (T=50%)": [
                generate_instance(n, "uncorrelated", 0.5, 1000 * n + s)
                for s in range(average_instances)
            ],
        }
        for scenario, instances in scenarios.items():
            for name in ("DFS", "B&B", "MITM", "DP"):
                times, ops = [], []
                for modules, T in instances:
                    # Large n makes the DFS slow, so it is timed once instead of three times.
                    t, r = measure(EXACT[name], modules, T, repetitions=3 if n <= 18 else 1)
                    times.append(t)
                    ops.append(r.operations)
                rows.append(
                    {
                        "n": n,
                        "scenario": scenario,
                        "algorithm": name,
                        "time_s": statistics.median(times),
                        "operations": statistics.mean(ops),
                    }
                )
        print(f"  n={n} ok")
    save_csv("exponential_time.csv", rows)

    _, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, scenario in zip(axes, ("worst case (T=100%)", "average case (T=50%)")):
        for name in ("DFS", "B&B", "MITM", "DP"):
            points = [
                (r["n"], r["operations"])
                for r in rows
                if r["scenario"] == scenario and r["algorithm"] == name
            ]
            ax.plot(*zip(*points), marker="o", label=name)
        if scenario.startswith("worst"):  # reference curve: the DFS visits at most 2^(n+1) nodes
            ax.plot(sizes, [2 ** (n + 1) for n in sizes], "k--", label="2^(n+1)")
        ax.set_yscale("log")
        ax.set_xlabel("n (modules)")
        ax.set_ylabel("operations (log)")
        ax.set_title(scenario)
        ax.legend()
        ax.grid(alpha=0.3)
    save_figure("fig_operations_vs_n.png")

    plt.figure(figsize=(6, 4))
    for name in ("DFS", "B&B", "MITM", "DP"):
        points = [
            (r["n"], r["time_s"])
            for r in rows
            if r["scenario"] == "worst case (T=100%)" and r["algorithm"] == name
        ]
        plt.plot(*zip(*points), marker="o", label=name)
    plt.yscale("log")
    plt.xlabel("n (modules)")
    plt.ylabel("time (s, log)")
    plt.title("Wall-clock time in the DFS worst case")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure("fig_time_vs_n.png")


def exp_scale(sizes):
    """Block ``scale``: DP and greedy for large n (the DFS cannot run here); confirms Theta(n*T)
    and Theta(n log n).
    """
    rows = []
    for n in sizes:
        modules, T = generate_instance(n, "uncorrelated", 0.5, 7 * n)
        for name, algorithm in {"DP": run_dp, "DP-vector": dp_vector, "Greedy": greedy}.items():
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
    for name in ("DP", "DP-vector", "Greedy"):
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


def exp_quality(ns, count, ratios=(0.1, 0.3, 0.5, 0.7), ref_n=15):
    """Block ``quality``: greedy and greedy+best-item gap from the optimum, by n, instance family
    and budget ratio.
    """
    rows = []
    for n in ns:
        for family in FAMILIES:
            for ratio in ratios:
                gaps_g, gaps_gi = [], []
                for s in range(count):
                    # A different seed per (n, ratio, repetition), reproducible across runs.
                    modules, T = generate_instance(n, family, ratio, 31 * s + int(100 * ratio) + n)
                    optimum = dp_vector(modules, T).value  # exact reference for the gap
                    gaps_g.append(gap(optimum, greedy(modules, T).value))
                    gaps_gi.append(gap(optimum, greedy_with_best_item(modules, T).value))
                rows.append(
                    {
                        "n": n,
                        "family": family,
                        "budget_ratio": ratio,
                        "greedy_mean_gap": statistics.mean(gaps_g),
                        "greedy_max_gap": max(gaps_g),
                        "greedy_item_mean_gap": statistics.mean(gaps_gi),
                        "greedy_item_max_gap": max(gaps_gi),
                    }
                )
    save_csv("greedy_quality.csv", rows)
    print("greedy mean gap / max gap (by budget ratio), instances per point:", count)
    for n in ns:
        for family in FAMILIES:
            cells = [r for r in rows if r["n"] == n and r["family"] == family]
            text = "  ".join(
                f"{r['budget_ratio']:.0%}: {r['greedy_mean_gap']:.1%}/{r['greedy_max_gap']:.1%}"
                for r in cells
            )
            print(f"  n={n:<3}{family:<22}{text}")

    _, axes = plt.subplots(1, 2, figsize=(11, 4))
    for family in FAMILIES:
        points = [
            (r["budget_ratio"] * 100, r["greedy_mean_gap"] * 100)
            for r in rows
            if r["family"] == family and r["n"] == ref_n
        ]
        axes[0].plot(*zip(*points), marker="o", label=family)
        points = [
            (r["n"], r["greedy_mean_gap"] * 100, r["greedy_max_gap"] * 100)
            for r in rows
            if r["family"] == family and r["budget_ratio"] == 0.3
        ]
        (line,) = axes[1].plot(
            [p[0] for p in points], [p[1] for p in points], marker="o", label=family + " (mean)"
        )
        axes[1].plot(
            [p[0] for p in points],
            [p[2] for p in points],
            "--",
            color=line.get_color(),
            label=family + " (max)",
        )
    axes[0].set_xlabel("budget T (% of the sum of times)")
    axes[0].set_ylabel("greedy mean gap (%)")
    axes[0].set_title(f"Gap by budget (n={ref_n})")
    axes[1].set_xlabel("n (modules)")
    axes[1].set_ylabel("greedy gap (%)")
    axes[1].set_title("Gap by n (T = 30% of the sum of times)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    save_figure("fig_greedy_quality.png")


def exp_pseudopolynomial(n, factors):
    """Block ``pseudo``: same modules with all times (and T) multiplied by a factor; DP cost grows
    with T, showing it is pseudo-polynomial.
    """
    base, T0 = generate_instance(n, "uncorrelated", 0.5, 99)
    rows = []
    for factor in factors:
        # Scaling every time and T by the same factor keeps the optimum identical but makes
        # the table T times wider: only the DP (and B&B/MITM through T) feel it.
        modules = [LearningModule(m.name, m.time * factor, m.score) for m in base]
        T = T0 * factor
        for name, algorithm in {
            "DFS": exhaustive_dfs,
            "B&B": branch_and_bound,
            "MITM": meet_in_the_middle,
            "DP": run_dp,
            "Greedy": greedy,
        }.items():
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
    for name in ("DFS", "B&B", "MITM", "DP", "Greedy"):
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
    """Block ``families``: operations of B&B, MITM and DP by instance family (strongly correlated
    instances hurt B&B the most).
    """
    rows = []
    for family in ("uncorrelated", "strongly_correlated"):
        for n in ns:
            ops_bb, ops_mitm, ops_dp = [], [], []
            for s in range(count):
                modules, T = generate_instance(n, family, 0.5, 17 * s + n, max_time)
                ops_bb.append(branch_and_bound(modules, T).operations)
                ops_mitm.append(meet_in_the_middle(modules, T).operations)
                ops_dp.append(dp_vector(modules, T).operations)
            rows.append(
                {
                    "family": family,
                    "n": n,
                    "bb_mean": statistics.mean(ops_bb),
                    "bb_max": max(ops_bb),
                    "mitm_mean": statistics.mean(ops_mitm),
                    "dp_mean": statistics.mean(ops_dp),
                }
            )
            r = rows[-1]
            print(
                f"  {family:<20} n={n:<3} B&B mean={r['bb_mean']:>11.0f} max={r['bb_max']:>11}"
                f"  MITM={r['mitm_mean']:>10.0f}  DP={r['dp_mean']:>9.0f}"
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
    plt.plot(
        [r["n"] for r in cells], [r["mitm_mean"] for r in cells], "g-^", label="MITM (any family)"
    )
    plt.yscale("log")
    plt.xlabel("n (modules)")
    plt.ylabel("operations (log)")
    plt.title(f"Exact algorithms by family (max time per module = {max_time})")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    save_figure("fig_exact_by_family.png")


def exp_fptas(n, epsilons):
    """Block ``fptas``: quality and cost of the FPTAS for several epsilons against the exact DP."""
    modules, T = generate_instance(n, "strongly_correlated", 0.5, 5)
    optimum = run_dp(modules, T)  # the FPTAS is judged against the exact value
    rows = [
        {
            "algorithm": "exact DP",
            "epsilon": 0,
            "value": optimum.value,
            "ratio_to_optimum": 1.0,
            "operations": optimum.operations,
        }
    ]
    for eps in epsilons:
        r = fptas(modules, T, eps)
        rows.append(
            {
                "algorithm": "FPTAS",
                "epsilon": eps,
                "value": r.value,
                "ratio_to_optimum": r.value / optimum.value,
                "operations": r.operations,
            }
        )
    save_csv("fptas.csv", rows)
    for r in rows:
        print(
            f"  {r['algorithm']:<9} eps={r['epsilon']:<5} value={r['value']:<5} "
            f"ratio={r['ratio_to_optimum']:.4f}  ops={r['operations']}"
        )


def exp_graph():
    """Block ``graph``: the paper's DFS on a complete DAG generates exactly 2^(n-1) paths."""
    rows = []
    for n in range(4, 17, 2):
        # Unit times and a huge budget: no pruning, so the DFS must list every path.
        _, _, paths = dfs_paths(complete_dag(n), [1] * n, [1] * n, 0, 10**6)
        rows.append({"n": n, "paths": paths, "2^(n-1)": 2 ** (n - 1)})
        print(f"  n={n:<3} paths generated={paths:<7} 2^(n-1)={2 ** (n - 1)}")
        assert paths == 2 ** (n - 1)
    save_csv("graph_dfs.csv", rows)


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
        help="blocks: demo correctness time scale quality pseudo families fptas graph",
    )
    args = parser.parse_args()
    q = args.quick  # --quick shrinks sizes and repetitions (about 5 s instead of minutes)
    blocks = {
        "demo": exp_demo,
        "correctness": lambda: exp_correctness(300 if q else 1200),
        "time": lambda: exp_exponential_time(
            list(range(10, 19, 2)) if q else list(range(10, 23, 2))
        ),
        "scale": lambda: exp_scale([50, 100, 200] if q else [50, 100, 200, 400, 600]),
        "quality": lambda: exp_quality([8, 15, 30] if q else [8, 15, 30, 60], 10 if q else 30),
        "pseudo": lambda: exp_pseudopolynomial(
            16 if q else 18, [1, 4, 16, 64] if q else [1, 4, 16, 64, 256, 1024]
        ),
        "families": lambda: exp_families([20, 30] if q else [20, 30, 40], 3 if q else 5),
        "fptas": lambda: exp_fptas(20 if q else 30, [0.5, 0.2, 0.1, 0.05]),
        "graph": exp_graph,
    }
    for name in args.only or blocks:
        print(f"\n### block: {name}")
        start = time.perf_counter()
        blocks[name]()
        print(f"   ({time.perf_counter() - start:.1f} s)")


if __name__ == "__main__":
    main()
