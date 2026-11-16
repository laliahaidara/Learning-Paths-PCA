"""Branch and bound with a fractional (Dantzig) upper bound (Dantzig, 1957).

Explores the "take / skip" tree depth first, but a branch is cut as soon as an
optimistic estimate of its best possible score cannot beat the best solution
found so far. Still exponential in the worst case, usually much faster.
"""

from .model import Result
from .sorting import sort_by_ratio


def branch_and_bound(modules, budget):
    """Optimal selection; explores modules in decreasing score/time order.

    ``operations`` = visited nodes + steps spent computing bounds + the
    comparisons of the initial sort.
    """
    n = len(modules)
    counter = [0]
    order = sort_by_ratio(modules, counter)  # best ratios first, so good solutions appear early
    best_value = 0
    best_indices = []
    chosen = []
    nodes = 0
    bound_steps = 0

    def upper_bound(k, used_time, score):
        """Optimistic score of a branch: fill the room greedily and allow the
        last module that does not fit to be taken fractionally (rounded down).
        No integer solution in this branch can exceed it."""
        nonlocal bound_steps
        room, ub = budget - used_time, score
        for j in range(k, n):
            bound_steps += 1
            m = modules[order[j]]
            if m.time <= room:
                room -= m.time
                ub += m.score
            else:
                return ub + (room * m.score) // m.time
        return ub

    def explore(k, used_time, score):
        nonlocal best_value, best_indices, nodes
        nodes += 1
        if score > best_value:
            best_value = score
            best_indices = list(chosen)
        # Prune: nothing left to decide, or the bound cannot beat the incumbent.
        if k == n or upper_bound(k, used_time, score) <= best_value:
            return
        i = order[k]
        if used_time + modules[i].time <= budget:  # take module i
            chosen.append(i)
            explore(k + 1, used_time + modules[i].time, score + modules[i].score)
            chosen.pop()
        explore(k + 1, used_time, score)  # skip module i

    explore(0, 0, 0)
    return Result("B&B", best_value, sorted(best_indices), nodes + bound_steps + counter[0])
