"""Validation of our hand-written algorithms against independent libraries.

Blocks (choose with --only): optimum, sorting, pearson, runtime, bnb.
Needs ortools, scipy and numpy. Output: results/validation_*.csv.

    python experiment_validation.py                  # every block (~10 min)
    python experiment_validation.py --only optimum   # one block
"""

import argparse
import random
import statistics
import time
from fractions import Fraction

import numpy as np
from ortools.algorithms.python import knapsack_solver
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp

from algorithms import (
    branch_and_bound,
    dp_memoized,
    dp_table,
    dp_vector,
    evaluate,
)
from algorithms.sorting import sort_by_ratio
from common import measure, save_csv
from instances import FAMILIES, generate_instance
from real_instances import (
    SCORE_DEFINITIONS,
    TOPIC_SCORE_DEFINITIONS,
    real_instance,
    real_topic_instance,
)


def ortools_optimum(modules, budget):
    """Optimum score according to Google OR-Tools' knapsack solver (the oracle)."""
    if not modules:
        return 0
    # OR-Tools offers two exact solvers: its branch and bound is fine up to ~40 items, and its
    # dynamic-programming solver is used beyond that, where B&B would be too slow.
    kind = (
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER
        if len(modules) <= 40
        else knapsack_solver.SolverType.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER
    )
    solver = knapsack_solver.KnapsackSolver(kind, "validation")
    solver.init([m.score for m in modules], [[m.time for m in modules]], [budget])
    return int(solver.solve())


def milp_optimum(modules, budget):
    """Optimum score according to SciPy/HiGHS, solving the knapsack as a MILP (second oracle)."""
    if not modules:
        return 0
    # One binary variable x_i per module; maximize sum(score_i * x_i) subject to
    # sum(time_i * x_i) <= budget. HiGHS minimizes, hence the negative objective c = -values.
    values = np.array([m.score for m in modules], dtype=float)
    weights = np.array([[m.time for m in modules]], dtype=float)
    result = milp(
        c=-values,
        constraints=LinearConstraint(weights, ub=budget),
        integrality=np.ones(len(modules)),
        bounds=Bounds(0, 1),
        options={"mip_rel_gap": 0.0},  # a gap of 0 demands a proven optimum, not an approximation
    )
    if result.x is None:
        raise RuntimeError(f"milp failed: {result.message}")
    chosen = [i for i, x in enumerate(result.x) if x > 0.5]  # round the 0/1 solution safely
    used, score = evaluate(modules, chosen)
    assert used <= budget, "milp returned an infeasible selection"
    return score


def protocol_instances(count=1200, seed=2024):
    """The 1,200 random instances of the correctness protocol (n <= 16, all families)."""
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(0, 16)
        yield generate_instance(
            n,
            rng.choice(FAMILIES),
            rng.choice([0.1, 0.3, 0.5, 0.7, 1.0]),
            rng.randrange(10**9),
            rng.choice([5, 20]),
        )


def grid_instances():
    """1,800 small instances (n = 8, 12, 16): families x seeds x budget ratios."""
    for family in FAMILIES:
        for seed in range(40):
            for rho in (0.1, 0.3, 0.5, 0.7, 1.0):
                for n in (8, 12, 16):
                    yield generate_instance(n, family, rho, seed * 1000 + n)


def large_instances():
    """Large synthetic instances (n = 50..600), where only the DP variants are run."""
    for n in (50, 100, 200, 400, 600):
        for family in FAMILIES:
            for rho in (0.3, 0.5):
                for seed in range(3):
                    yield generate_instance(n, family, rho, 7000 + 13 * n + seed)


def real_module_instances():
    """Real module-level instances (12 modules) for budgets from 30 min to 24 h."""
    for score in SCORE_DEFINITIONS:
        modules = real_instance(score=score)
        for budget in range(30, 24 * 60 + 1, 15):
            yield modules, budget


def real_topic_instances():
    """Real topic-level instances (78 topics) for budgets from 30 min to 20 h."""
    for score in TOPIC_SCORE_DEFINITIONS:
        modules = real_topic_instance(score=score)
        for budget in range(30, 20 * 60 + 1, 15):
            yield modules, budget


# Which hand-written exact algorithms run on which instances: B&B can be too slow for
# n >= 50 on strongly correlated data, so the large instances only run the DP variants (and
# B&B for the 78 topics, which finishes there).
SMALL_EXACT = {
    "DP (memoizada)": dp_memoized,
    "DP (vetor)": dp_vector,
    "B&B": branch_and_bound,
}
LARGE_EXACT = {"DP (memoizada)": dp_memoized, "DP (vetor)": dp_vector}
TOPIC_EXACT = {"DP (memoizada)": dp_memoized, "DP (vetor)": dp_vector, "B&B": branch_and_bound}


def exp_optimum():
    """Block ``optimum``: every exact algorithm must match OR-Tools and SciPy/HiGHS on every
    instance.
    """
    print("== Optimum: hand-written exact algorithms vs OR-Tools and SciPy/HiGHS")
    sets = [
        ("protocolo de correção (n<=16)", protocol_instances(), SMALL_EXACT),
        ("grade de instâncias pequenas (n<=16)", grid_instances(), SMALL_EXACT),
        ("sintéticas grandes (n=50..600)", large_instances(), LARGE_EXACT),
        ("reais, módulos (n=12)", real_module_instances(), SMALL_EXACT),
        ("reais, tópicos (n=78)", real_topic_instances(), TOPIC_EXACT),
    ]
    rows = []
    for label, instances, extra in sets:
        # Counters per instance set: total, disagreements between the two oracles, infeasible DP
        # selections, and how many instances each algorithm gets wrong.
        total = oracle_disagree = infeasible = 0
        mismatches = {name: 0 for name in ["DP (tabela)"] + list(extra)}
        worst = 0
        for modules, budget in instances:
            total += 1
            reference = ortools_optimum(modules, budget)  # first oracle
            # Second, independent oracle: if the two disagree, neither is trusted for this instance.
            if milp_optimum(modules, budget) != reference:
                oracle_disagree += 1
            table = dp_table(modules, budget)[0]
            # The reconstructed path of the DP table must fit in the budget and add up to its value.
            used, score = evaluate(modules, table.indices)
            if used > budget or score != table.value:
                infeasible += 1
            results = {"DP (tabela)": table.value}
            results.update(
                {name: algorithm(modules, budget).value for name, algorithm in extra.items()}
            )
            for name, value in results.items():
                if value != reference:  # any difference from the oracle is a divergence
                    mismatches[name] += 1
                    worst = max(worst, abs(value - reference))
        row = {
            "conjunto": label,
            "instancias": total,
            "oraculos_divergem": oracle_disagree,
            "selecao_dp_inviavel": infeasible,
            "maior_diferenca": worst,
        }
        row.update(
            {
                f"divergencias {name}": mismatches.get(name, "")
                for name in ["DP (tabela)", "DP (memoizada)", "DP (vetor)", "B&B"]
            }
        )
        rows.append(row)
        print(
            f"  {label:32s} {total:5d} instâncias  divergências: "
            f"{sum(mismatches.values())}  (oráculos entre si: {oracle_disagree})"
        )
    save_csv("validation_optimum.csv", rows)


def exp_sorting():
    """Block ``sorting``: our merge sort gives the same order as sorted() with exact Fraction keys.
    """
    print("== Merge sort vs sorted() (exact Fraction keys, both stable)")
    total = different = 0
    sources = list(protocol_instances()) + list(large_instances())
    sources += [(real_instance(score=s), 0) for s in SCORE_DEFINITIONS]
    sources += [(real_topic_instance(score=s), 0) for s in TOPIC_SCORE_DEFINITIONS]
    for modules, _ in sources:
        total += 1
        mine = sort_by_ratio(modules, [0])  # [0] is a throw-away comparison counter
        # Fraction keys compare ratios exactly; reverse=True with a stable sort keeps tie order.
        reference = sorted(
            range(len(modules)),
            key=lambda i: Fraction(modules[i].score, modules[i].time),
            reverse=True,
        )
        different += mine != reference
    print(f"  {total} listas ordenadas, ordens diferentes: {different}")
    save_csv("validation_sorting.csv", [{"listas": total, "ordens_diferentes": different}])


def exp_pearson():
    """Block ``pearson``: time-score correlation computed with statistics, NumPy and SciPy must
    agree.
    """
    print("== Pearson: statistics.correlation vs numpy.corrcoef vs scipy.stats.pearsonr")
    rows = []
    instances = [("modulos", s, real_instance(score=s)) for s in SCORE_DEFINITIONS]
    instances += [("topicos", s, real_topic_instance(score=s)) for s in TOPIC_SCORE_DEFINITIONS]
    for level, score, modules in instances:
        xs = [m.time for m in modules]
        ys = [m.score for m in modules]
        own = statistics.correlation(xs, ys)  # the implementation used in the report
        with_numpy = float(np.corrcoef(xs, ys)[0, 1])
        result = stats.pearsonr(xs, ys)
        low, high = result.confidence_interval(0.95)
        rows.append(
            {
                "instancia": level,
                "score": score,
                "n": len(modules),
                "r_relatorio": round(own, 4),
                "dif_numpy": f"{abs(own - with_numpy):.1e}",
                "dif_scipy": f"{abs(own - result.statistic):.1e}",
                "p_valor": round(result.pvalue, 4),
                "ic95_inf": round(low, 2),
                "ic95_sup": round(high, 2),
            }
        )
        print(
            f"  {level}/{score:12s} n={len(modules):2d}  r={own:+.3f}  p={result.pvalue:.4f}  "
            f"IC95=[{low:+.2f}, {high:+.2f}]  "
            f"dif. máx={max(abs(own - with_numpy), abs(own - result.statistic)):.1e}"
        )
    save_csv("validation_pearson.csv", rows)


def exp_runtime():
    """Block ``runtime``: pure-Python DP against OR-Tools (C++) and SciPy/HiGHS."""
    print("== Runtime: DP in pure Python vs OR-Tools (C++) vs SciPy/HiGHS")
    rows = []
    for n in (50, 100, 200, 400, 600):
        modules, budget = generate_instance(n, "weakly_correlated", 0.5, 7000 + n)
        t_dp, _ = measure(dp_table, modules, budget)
        t_or, _ = measure(ortools_optimum, modules, budget)
        t_milp, _ = measure(milp_optimum, modules, budget)
        rows.append(
            {
                "n": n,
                "T": budget,
                "dp_python_ms": round(1000 * t_dp, 2),
                "ortools_ms": round(1000 * t_or, 3),
                "scipy_milp_ms": round(1000 * t_milp, 2),
                "dp_sobre_ortools": round(t_dp / t_or, 1),
                "dp_sobre_milp": round(t_dp / t_milp, 2),
            }
        )
        print(
            f"  n={n:3d} T={budget:5d}  DP {1000 * t_dp:8.2f} ms  OR-Tools {1000 * t_or:7.3f} ms  "
            f"HiGHS {1000 * t_milp:7.2f} ms"
        )
    save_csv("validation_runtime.csv", rows)


def _bnb_worker(n, seed, queue):
    """Run our B&B in a child process and report (value, operations, seconds) through ``queue``."""
    modules, budget = generate_instance(n, "strongly_correlated", 0.5, seed)
    start = time.perf_counter()
    result = branch_and_bound(modules, budget)
    queue.put((result.value, result.operations, time.perf_counter() - start))


def exp_bnb(limit_seconds=20):
    """Block ``bnb``: does our B&B finish on large strongly correlated instances within the time
    limit?
    """
    import multiprocessing

    print(f"== Hand-written B&B on strongly correlated instances (limit {limit_seconds} s per run)")
    rows = []
    for n in (50, 100, 200, 400, 600):
        for k in range(3):
            seed = 7000 + 13 * n + k
            modules, budget = generate_instance(n, "strongly_correlated", 0.5, seed)
            reference = ortools_optimum(modules, budget)
            t_dp, _ = measure(dp_table, modules, budget, repetitions=1)
            # B&B runs in a child process so it can be killed when it exceeds the time limit.
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=_bnb_worker, args=(n, seed, queue))
            process.start()
            process.join(limit_seconds)
            if process.is_alive():
                process.terminate()
                process.join()
                outcome = {
                    "bnb_terminou": "não",
                    "bnb_s": f">{limit_seconds}",
                    "bnb_operacoes": "",
                    "bnb_correto": "",
                }
            else:
                value, operations, seconds = queue.get()
                outcome = {
                    "bnb_terminou": "sim",
                    "bnb_s": round(seconds, 3),
                    "bnb_operacoes": operations,
                    "bnb_correto": "sim" if value == reference else "NÃO",
                }
            rows.append({"n": n, "T": budget, "semente": seed, **outcome, "dp_s": round(t_dp, 3)})
            print(
                f"  n={n:3d} T={budget:5d}  B&B: {outcome['bnb_terminou']:3s} "
                f"{outcome['bnb_s']} s   DP: {t_dp:.3f} s"
            )
    save_csv("validation_bnb.csv", rows)


# Block name -> function; ``--only`` picks from these keys.
BLOCKS = {
    "optimum": exp_optimum,
    "sorting": exp_sorting,
    "pearson": exp_pearson,
    "runtime": exp_runtime,
    "bnb": exp_bnb,
}


def main():
    """Parse the command line and run the selected blocks."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="*", choices=sorted(BLOCKS), default=None)
    args = parser.parse_args()
    for name, block in BLOCKS.items():
        if args.only is None or name in args.only:
            block()


if __name__ == "__main__":
    main()
