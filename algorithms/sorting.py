def merge(left, right, comes_before, counter):
    output = []
    i = j = 0
    while i < len(left) and j < len(right):
        counter[0] += 1
        if comes_before(right[j], left[i]):
            output.append(right[j])
            j += 1
        else:
            output.append(left[i])
            i += 1
    output.extend(left[i:])
    output.extend(right[j:])
    return output


def merge_sort(items, comes_before, counter):
    if len(items) <= 1:
        return list(items)
    middle = len(items) // 2
    left = merge_sort(items[:middle], comes_before, counter)
    right = merge_sort(items[middle:], comes_before, counter)
    return merge(left, right, comes_before, counter)


def sort_by_ratio(modules, counter):
    def ratio_greater(i, j):
        return modules[i].score * modules[j].time > modules[j].score * modules[i].time

    return merge_sort(list(range(len(modules))), ratio_greater, counter)
