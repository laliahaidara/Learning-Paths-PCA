"""Exhaustive depth-first search (the paper's RUTICO idea, knapsack version).

At each module the search branches in two: take it (if it still fits) or skip
it. Every feasible subset is visited, so the cost is Theta(2^n) in the worst
case. It is the reference ("ground truth") used to check all other algorithms.
"""

from .model import Result


def exhaustive_dfs(modules, budget):
    """Optimal selection by trying every feasible subset.

    Time: O(2^n) nodes. Space: O(n) (recursion depth and the current path).
    ``operations`` counts the visited nodes of the search tree.
    """
    n = len(modules)
    best_value = 0
    best_indices = []
    chosen = []  # the current path (modules taken so far)
    nodes = 0

    def explore(i, used_time, score):
        # State: decisions made for modules 0..i-1; `used_time`/`score` are the
        # totals of `chosen`.
        nonlocal best_value, best_indices, nodes
        nodes += 1
        if score > best_value:  # every node is itself a complete feasible subset
            best_value = score
            best_indices = list(chosen)
        if i == n:
            return
        w = modules[i].time
        if used_time + w <= budget:  # branch 1: take module i (pruned if it does not fit)
            chosen.append(i)
            explore(i + 1, used_time + w, score + modules[i].score)
            chosen.pop()  # backtrack
        explore(i + 1, used_time, score)  # branch 2: skip module i

    explore(0, 0, 0)
    return Result("DFS", best_value, best_indices, nodes)
