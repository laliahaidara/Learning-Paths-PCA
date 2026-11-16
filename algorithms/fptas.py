"""FPTAS: a (1 - epsilon) approximation in O(n^3 / epsilon) time.

Idea: the exact DP is pseudo-polynomial in the *time* budget. Here the roles
are swapped: scores are scaled down (rounded) so that their sum is small, and
the DP is run over scores, storing the minimum time needed to reach each
(scaled) score. Rounding loses at most epsilon * OPT, so the result is at
least (1 - epsilon) * OPT, while the table size depends on n and 1/epsilon
only, not on T.
"""

from .model import Result


def fptas(modules, budget, epsilon):
    """Approximate selection with value >= (1 - epsilon) * optimum."""
    # Modules that can never be chosen (do not fit alone, or are worth nothing) are ignored.
    cand = [i for i, m in enumerate(modules) if m.time <= budget and m.score > 0]
    if not cand:
        return Result(f"FPTAS({epsilon})", 0, [], 0)
    n = len(cand)
    # Scaling factor K: scores become floor(score / K), so the largest
    # scaled score is about n / epsilon and the scaled total S is O(n^2 / epsilon).
    K = epsilon * max(modules[i].score for i in cand) / n
    scaled = [int(modules[i].score / K) for i in cand]
    S = sum(scaled)
    inf = budget + 1  # "unreachable": any real time is <= budget
    # weight[k][s] = minimum time to reach scaled score exactly s with the first k candidates.
    weight = [[inf] * (S + 1) for _ in range(n + 1)]
    weight[0][0] = 0
    operations = 0
    for k in range(1, n + 1):
        w, e = modules[cand[k - 1]].time, scaled[k - 1]
        for s in range(S + 1):
            operations += 1
            weight[k][s] = weight[k - 1][s]
            new_time = weight[k - 1][s - e] + w if s >= e else inf
            if new_time <= budget and new_time < weight[k][s]:
                weight[k][s] = new_time

    # Largest scaled score reachable within the time budget, then walk back to recover the modules.
    s = max(x for x in range(S + 1) if weight[n][x] <= budget)
    chosen = []
    for k in range(n, 0, -1):
        if weight[k][s] != weight[k - 1][s]:
            chosen.append(cand[k - 1])
            s -= scaled[k - 1]
    value = sum(modules[i].score for i in chosen)  # real (unscaled) score of the chosen set
    return Result(f"FPTAS({epsilon})", value, sorted(chosen), operations)
