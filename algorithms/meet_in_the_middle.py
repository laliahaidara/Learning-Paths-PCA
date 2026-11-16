from .model import Result
from .sorting import merge


def _subsets_by_time(modules, indices, budget, counter):
    subsets = [(0, 0, 0)]
    for bit, i in enumerate(indices):
        w, v = modules[i].time, modules[i].score
        counter[0] += len(subsets)
        shifted = [(t + w, s + v, mask | (1 << bit)) for (t, s, mask) in subsets if t + w <= budget]
        subsets = merge(subsets, shifted, lambda a, b: a[0] < b[0], counter)
    return subsets


def _decode(mask, indices):
    return [indices[b] for b in range(len(indices)) if (mask >> b) & 1]


def meet_in_the_middle(modules, budget):
    n = len(modules)
    half = n // 2
    idx_a, idx_b = list(range(half)), list(range(half, n))
    counter = [0]
    A = _subsets_by_time(modules, idx_a, budget, counter)
    B = _subsets_by_time(modules, idx_b, budget, counter)

    best_up_to = []
    for j, (_, score, _) in enumerate(B):
        if j == 0 or score > B[best_up_to[-1]][1]:
            best_up_to.append(j)
        else:
            best_up_to.append(best_up_to[-1])

    best_value, pair = -1, (0, 0)
    j = len(B) - 1
    for a in A:
        while B[j][0] + a[0] > budget:
            j -= 1
            counter[0] += 1
        counter[0] += 1
        k = best_up_to[j]
        if a[1] + B[k][1] > best_value:
            best_value, pair = a[1] + B[k][1], (a[2], B[k][2])

    indices = _decode(pair[0], idx_a) + _decode(pair[1], idx_b)
    return Result("MITM", best_value, indices, counter[0])
