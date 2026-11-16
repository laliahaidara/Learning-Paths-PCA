from .model import Result
from .sorting import sort_by_ratio


def branch_and_bound(modules, budget):
    n = len(modules)
    counter = [0]
    order = sort_by_ratio(modules, counter)
    best_value = 0
    best_indices = []
    chosen = []
    nodes = 0
    bound_steps = 0

    def upper_bound(k, used_time, score):
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
        if k == n or upper_bound(k, used_time, score) <= best_value:
            return
        i = order[k]
        if used_time + modules[i].time <= budget:
            chosen.append(i)
            explore(k + 1, used_time + modules[i].time, score + modules[i].score)
            chosen.pop()
        explore(k + 1, used_time, score)

    explore(0, 0, 0)
    return Result("B&B", best_value, sorted(best_indices), nodes + bound_steps + counter[0])
