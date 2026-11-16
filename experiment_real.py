"""Experiments on the real (anonymized) platform instances.

Usage:
    python experiment_real.py                   (all blocks)
    python experiment_real.py --only profile    (one block)

Blocks:
    profile      - dataset description and Pearson correlation time x score,
                   locating the real data among the synthetic families;
    algorithms   - DP and B&B on the real catalog, per weekly budget;
    pseudopoly   - same instance in hours / 15-min blocks / minutes (T grows,
                   n fixed): the practical price of the DP's pseudo-polynomiality;
    sweep        - one DP table answers every budget (last row), per score.
"""
import argparse
import statistics

from algorithms import branch_and_bound, dp_memoized, dp_table, dp_vector
from common import run_dp, save_csv
from instances import generate_instance
from real_instances import (SCORE_DEFINITIONS, TOPIC_SCORE_DEFINITIONS,
                            real_instance, real_topic_instance,
                            weekly_budgets_minutes)


ALL = {"DP": run_dp, "DP-vector": dp_vector, "DP-memo": dp_memoized,
       "B&B": branch_and_bound}
EXACT_REFERENCE = "DP"


def pearson(xs, ys):
    return statistics.correlation(xs, ys)


# ------------------------------------------------------------------ profile
def exp_profile():
    """Where does the real data sit among the synthetic families?"""
    print("== Real-data profile (times in minutes)")
    rows = []
    for score_def in SCORE_DEFINITIONS:
        modules = real_instance(score=score_def)
        times = [m.time for m in modules]
        scores = [m.score for m in modules]
        try:
            r = pearson(times, scores)
        except statistics.StatisticsError:
            r = float("nan")
        rows.append({"score": score_def, "n": len(modules),
                     "min_time": min(times), "max_time": max(times),
                     "pearson_time_score": round(r, 3)})
        print(f"  {score_def:10s} n={len(modules):2d} "
              f"tempos {min(times)}..{max(times)} min, Pearson r={r:+.3f}")
    # Synthetic references at the same n, for the table in the paper
    for family in ("uncorrelated", "weakly_correlated", "strongly_correlated"):
        sample_r = []
        for seed in range(30):
            modules, _ = generate_instance(12, family, 0.5, seed, max_time=400)
            sample_r.append(pearson([m.time for m in modules],
                                    [m.score for m in modules]))
        mean_r = statistics.mean(sample_r)
        rows.append({"score": f"sintetica:{family}", "n": 12, "min_time": "",
                     "max_time": "", "pearson_time_score": round(mean_r, 3)})
        print(f"  {family:22s} (sintetica, media de 30) r={mean_r:+.3f}")
    save_csv("real_profile.csv", rows)


# --------------------------------------------------------------- algorithms
def exp_algorithms():
    """DP and B&B on the real catalog, one line per (score, budget)."""
    print("== DP and B&B on the real catalog")
    rows = []
    for score_def in SCORE_DEFINITIONS:
        modules = real_instance(score=score_def)
        for label, budget in weekly_budgets_minutes().items():
            optimum = run_dp(modules, budget).value
            for name, algorithm in ALL.items():
                result = algorithm(modules, budget)
                assert result.value == optimum, f"{name} is not optimal"
                rows.append({"score": score_def, "orcamento": label,
                             "algoritmo": name, "valor": result.value,
                             "otimo": optimum, "operacoes": result.operations})
            dp_row = [r for r in rows if r["score"] == score_def
                      and r["orcamento"] == label and r["algoritmo"] == "DP"][0]
            bb_row = [r for r in rows if r["score"] == score_def
                      and r["orcamento"] == label and r["algoritmo"] == "B&B"][0]
            print(f"  {score_def:10s} T={label:3s} otimo={optimum:4d} "
                  f"ops DP={dp_row['operacoes']} B&B={bb_row['operacoes']}")
    save_csv("real_all_algorithms.csv", rows)


# --------------------------------------------------------------- pseudopoly
def exp_pseudopoly():
    """n fixed, finer time unit => larger T: only the DP pays for it."""
    print("== Pseudo-polynomial behavior on the real instance (score=conteudo)")
    rows = []
    for unit, label in ((60, "horas"), (15, "blocos de 15 min"), (5, "blocos de 5 min"),
                        (1, "minutos")):
        modules = real_instance(score="conteudo", unit_minutes=unit)
        budget = max(1, round(600 / unit))  # 10 h in the chosen unit
        for name in ("DP", "B&B"):
            result = ALL[name](modules, budget)
            rows.append({"unidade": label, "T": budget, "algoritmo": name,
                         "valor": result.value, "operacoes": result.operations})
        dp_ops = [r for r in rows if r["unidade"] == label and r["algoritmo"] == "DP"][0]
        print(f"  {label:18s} T={budget:4d}: DP {dp_ops['operacoes']:6d} ops")
    save_csv("real_pseudopolynomial.csv", rows)


# -------------------------------------------------------------------- sweep
def exp_sweep():
    """One DP table gives the optimal path value for EVERY budget up to 24h."""
    print("== Budget sweep from a single DP table (last row)")
    rows = []
    horizon = 24 * 60
    for score_def in SCORE_DEFINITIONS:
        modules = real_instance(score=score_def)
        _, table = dp_table(modules, horizon)
        last = table[len(modules)]
        for budget in range(0, horizon + 1, 15):
            rows.append({"score": score_def, "T_min": budget, "otimo": last[budget]})
        full = sum(m.time for m in modules)
        print(f"  {score_def:10s} otimo com 2h={last[120]}, 5h={last[300]}, "
              f"10h={last[600]}, tudo cabe a partir de {full} min")
    save_csv("real_budget_sweep.csv", rows)


# ------------------------------------------------------------- topic level
# At n=78 an exhaustive enumeration is infeasible by design (2^78 subsets):
# exactly the regime where only DP and B&B apply.
TOPIC_ALGORITHMS = {"DP": run_dp, "DP-vector": dp_vector, "DP-memo": dp_memoized,
                    "B&B": branch_and_bound}


def exp_topic_profile():
    print("== Topic-level profile (n grows, times from activity-weighted split)")
    rows = []
    for score_def in TOPIC_SCORE_DEFINITIONS:
        modules = real_topic_instance(score=score_def)
        times = [m.time for m in modules]
        scores = [m.score for m in modules]
        r = pearson(times, scores)
        rows.append({"score": score_def, "n": len(modules),
                     "min_time": min(times), "max_time": max(times),
                     "soma_tempos_min": sum(times),
                     "pearson_time_score": round(r, 3)})
        print(f"  {score_def:12s} n={len(modules)} tempos {min(times)}..{max(times)} min, "
              f"total {sum(times)} min, Pearson r={r:+.3f}")
    save_csv("real_topic_profile.csv", rows)


def exp_topic_algorithms():
    print("== Algorithms on the topic-level instance (n=78)")
    rows = []
    for score_def in TOPIC_SCORE_DEFINITIONS:
        modules = real_topic_instance(score=score_def)
        for label, budget in weekly_budgets_minutes().items():
            optimum = run_dp(modules, budget).value
            for name, algorithm in TOPIC_ALGORITHMS.items():
                result = algorithm(modules, budget)
                assert result.value == optimum, f"{name} is not optimal"
                rows.append({"score": score_def, "orcamento": label,
                             "algoritmo": name, "valor": result.value,
                             "otimo": optimum, "operacoes": result.operations})
            bb_row = [r for r in rows if r["score"] == score_def
                      and r["orcamento"] == label and r["algoritmo"] == "B&B"][0]
            dp_row = [r for r in rows if r["score"] == score_def
                      and r["orcamento"] == label and r["algoritmo"] == "DP"][0]
            print(f"  {score_def:12s} T={label:3s} otimo={optimum:6d} "
                  f"ops DP={dp_row['operacoes']} B&B={bb_row['operacoes']}")
    save_csv("real_topic_algorithms.csv", rows)


BLOCKS = {"profile": exp_profile, "algorithms": exp_algorithms,
          "pseudopoly": exp_pseudopoly, "sweep": exp_sweep,
          "tprofile": exp_topic_profile, "talgorithms": exp_topic_algorithms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(BLOCKS), default=None)
    args = parser.parse_args()
    for name, block in BLOCKS.items():
        if args.only is None or name in args.only:
            block()


if __name__ == "__main__":
    main()
