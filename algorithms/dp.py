"""Dynamic programming for the 0/1 knapsack: Theta(n * T) time (pseudo-polynomial).

Recurrence, with D[i][t] = best score using only the first i modules and at
most t units of time:

    D[0][t] = 0
    D[i][t] = D[i-1][t]                                   if w_i > t
    D[i][t] = max(D[i-1][t], D[i-1][t - w_i] + v_i)       if w_i <= t

The answer is D[n][T]. It is the DFS "with memory": the 2^n subsets collapse
into only n * (T + 1) distinct states (i, remaining time).
"""

import sys

from .model import Result


def dp_table(modules, budget):
    """Bottom-up DP that keeps the whole table so the chosen modules can be recovered.

    Time and space: Theta(n * T). Returns ``(Result, D)``; ``operations`` counts
    the filled cells, so it equals n * (T + 1).
    """
    n = len(modules)
    D = [[0] * (budget + 1) for _ in range(n + 1)]  # row 0 stays all zeros
    operations = 0
    for i in range(1, n + 1):
        w, v = modules[i - 1].time, modules[i - 1].score
        for t in range(budget + 1):
            operations += 1
            D[i][t] = D[i - 1][t]  # option 1: do not take module i
            if w <= t and D[i - 1][t - w] + v > D[i][t]:  # option 2: take it
                D[i][t] = D[i - 1][t - w] + v

    # Reconstruction: walk back from D[n][T]. If the value changed between rows
    # i-1 and i, module i was taken, so we give back its time and continue.
    chosen = []
    t = budget
    for i in range(n, 0, -1):
        if D[i][t] != D[i - 1][t]:
            chosen.append(i - 1)
            t -= modules[i - 1].time
    chosen.reverse()
    return Result("DP (table)", D[n][budget], chosen, operations), D


def dp_vector(modules, budget):
    """Space-optimized DP: a single vector of size T + 1 (Theta(T) memory).

    Walking t from high to low guarantees each module is used at most once
    (the cells still to be read have not been updated yet). The value is
    exact, but the chosen modules cannot be recovered, so ``indices`` is empty.
    """
    d = [0] * (budget + 1)
    operations = 0
    for m in modules:
        for t in range(budget, m.time - 1, -1):
            operations += 1
            if d[t - m.time] + m.score > d[t]:
                d[t] = d[t - m.time] + m.score
    return Result("DP (vector)", d[budget], [], operations)


def dp_memoized(modules, budget):
    """Top-down DP: the DFS recursion plus a cache of already solved states.

    Only the states (i, t) that the recursion actually reaches are stored, so
    it can be cheaper than the full table. ``operations`` is the number of
    cached states.
    """
    n = len(modules)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 100))
    memo = {}

    def best(i, t):
        """Best score using modules i..n-1 with t units of time left."""
        if i == n:
            return 0
        if (i, t) in memo:
            return memo[(i, t)]
        answer = best(i + 1, t)  # skip module i
        w = modules[i].time
        if w <= t:  # take module i
            answer = max(answer, best(i + 1, t - w) + modules[i].score)
        memo[(i, t)] = answer
        return answer

    value = best(0, budget)
    # Reconstruction: module i is in the optimum when taking it is what
    # produces best(i, t) (that is, the value drops if we skip it).
    chosen = []
    t = budget
    for i in range(n):
        if best(i, t) != best(i + 1, t):
            chosen.append(i)
            t -= modules[i].time
    return Result("DP (memoized)", value, chosen, len(memo))
