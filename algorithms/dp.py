import sys

from .model import Result


def dp_table(modules, budget):
    n = len(modules)
    D = [[0] * (budget + 1) for _ in range(n + 1)]
    operations = 0
    for i in range(1, n + 1):
        w, v = modules[i - 1].time, modules[i - 1].score
        for t in range(budget + 1):
            operations += 1
            D[i][t] = D[i - 1][t]
            if w <= t and D[i - 1][t - w] + v > D[i][t]:
                D[i][t] = D[i - 1][t - w] + v

    chosen = []
    t = budget
    for i in range(n, 0, -1):
        if D[i][t] != D[i - 1][t]:
            chosen.append(i - 1)
            t -= modules[i - 1].time
    chosen.reverse()
    return Result("DP (table)", D[n][budget], chosen, operations), D


def dp_vector(modules, budget):
    d = [0] * (budget + 1)
    operations = 0
    for m in modules:
        for t in range(budget, m.time - 1, -1):
            operations += 1
            if d[t - m.time] + m.score > d[t]:
                d[t] = d[t - m.time] + m.score
    return Result("DP (vector)", d[budget], [], operations)


def dp_memoized(modules, budget):
    n = len(modules)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 100))
    memo = {}

    def best(i, t):
        if i == n:
            return 0
        if (i, t) in memo:
            return memo[(i, t)]
        answer = best(i + 1, t)
        w = modules[i].time
        if w <= t:
            answer = max(answer, best(i + 1, t - w) + modules[i].score)
        memo[(i, t)] = answer
        return answer

    value = best(0, budget)
    chosen = []
    t = budget
    for i in range(n):
        if best(i, t) != best(i + 1, t):
            chosen.append(i)
            t -= modules[i].time
    return Result("DP (memoized)", value, chosen, len(memo))
