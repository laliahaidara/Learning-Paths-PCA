"""Hand-written merge sort, Theta(n log n), with a comparison counter.

Written by hand (instead of ``sorted``) so the cost of sorting is counted in
the ``operations`` of the branch and bound.
"""


def merge(left, right, comes_before, counter):
    """Merge two sorted lists into one sorted list (stable).

    ``comes_before(a, b)`` is True when ``a`` must be placed before ``b``;
    ``counter[0]`` is incremented once per comparison.
    """
    output = []
    i = j = 0
    while i < len(left) and j < len(right):
        counter[0] += 1
        if comes_before(right[j], left[i]):  # strict: ties keep the left element first
            output.append(right[j])
            j += 1
        else:
            output.append(left[i])
            i += 1
    output.extend(left[i:])
    output.extend(right[j:])
    return output


def merge_sort(items, comes_before, counter):
    """Sort ``items`` with merge sort; returns a new list. O(n log n) comparisons."""
    if len(items) <= 1:
        return list(items)
    middle = len(items) // 2
    left = merge_sort(items[:middle], comes_before, counter)
    right = merge_sort(items[middle:], comes_before, counter)
    return merge(left, right, comes_before, counter)


def sort_by_ratio(modules, counter):
    """Indices of ``modules`` ordered by decreasing score/time ratio.

    Ratios are compared by cross-multiplication (score_i * time_j versus
    score_j * time_i), so no floating point is involved and ties are exact.
    """

    def ratio_greater(i, j):
        return modules[i].score * modules[j].time > modules[j].score * modules[i].time

    return merge_sort(list(range(len(modules))), ratio_greater, counter)
