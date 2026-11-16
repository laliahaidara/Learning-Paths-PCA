"""Meet in the middle (Horowitz-Sahni): O(2^(n/2)) time, optimal.

Split the modules into two halves A and B, list the feasible subsets of each
half sorted by time, then pair them with two pointers: for each subset of A,
the best partner in B is the highest-score subset that still fits in the time
left. Much faster than 2^n, but memory also grows as 2^(n/2).
"""

from .model import Result
from .sorting import merge


def _subsets_by_time(modules, indices, budget, counter):
    """All subsets of ``indices`` with time <= budget, sorted by time.

    Each entry is (total time, total score, bitmask). Adding one module
    merges the old list with a shifted copy of itself, so the list stays
    sorted without ever calling a full sort.
    """
    subsets = [(0, 0, 0)]  # only the empty set
    for bit, i in enumerate(indices):
        w, v = modules[i].time, modules[i].score
        counter[0] += len(subsets)
        shifted = [(t + w, s + v, mask | (1 << bit)) for (t, s, mask) in subsets if t + w <= budget]
        subsets = merge(subsets, shifted, lambda a, b: a[0] < b[0], counter)
    return subsets


def _decode(mask, indices):
    """Turn a bitmask over ``indices`` back into the list of module indices."""
    return [indices[b] for b in range(len(indices)) if (mask >> b) & 1]


def meet_in_the_middle(modules, budget):
    """Optimal selection by combining the subsets of two halves of the modules."""
    n = len(modules)
    half = n // 2
    idx_a, idx_b = list(range(half)), list(range(half, n))
    counter = [0]
    A = _subsets_by_time(modules, idx_a, budget, counter)
    B = _subsets_by_time(modules, idx_b, budget, counter)

    # best_up_to[j] = position of the highest-score subset among B[0..j]
    # (B is sorted by time, so this is "best score using at most that time").
    best_up_to = []
    for j, (_, score, _) in enumerate(B):
        if j == 0 or score > B[best_up_to[-1]][1]:
            best_up_to.append(j)
        else:
            best_up_to.append(best_up_to[-1])

    # Two pointers: as the time of the A-subset grows, the largest compatible
    # B-subset can only move left, so the whole sweep is linear.
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
