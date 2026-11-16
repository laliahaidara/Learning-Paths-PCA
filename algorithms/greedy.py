"""Greedy heuristics: fast (Theta(n log n)) but not optimal.

Plain greedy can be arbitrarily bad (see the adversarial instance in
experiment.py); adding the best single module gives the classic 1/2
approximation guarantee.
"""

from .model import Result
from .sorting import sort_by_ratio


def greedy(modules, budget):
    """Take modules in decreasing score/time order while they still fit.

    Cost: the sort (n log n comparisons) plus one check per module.
    """
    counter = [0]
    order = sort_by_ratio(modules, counter)
    used_time = value = 0
    chosen = []
    for i in order:
        counter[0] += 1
        if used_time + modules[i].time <= budget:
            used_time += modules[i].time
            value += modules[i].score
            chosen.append(i)
    return Result("Greedy", value, sorted(chosen), counter[0])


def greedy_with_best_item(modules, budget):
    """Better of: the greedy solution, or the single highest-score module that fits.

    Guarantee: the result is at least half of the optimum (OPT / 2).
    """
    base = greedy(modules, budget)
    best = None
    for i, m in enumerate(modules):
        if m.time <= budget and (best is None or m.score > modules[best].score):
            best = i
    operations = base.operations + len(modules)
    if best is not None and modules[best].score > base.value:
        return Result("Greedy+item", modules[best].score, [best], operations)
    return Result("Greedy+item", base.value, base.indices, operations)
