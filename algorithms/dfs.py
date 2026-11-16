from .model import Result


def exhaustive_dfs(modules, budget):
    n = len(modules)
    best_value = 0
    best_indices = []
    chosen = []
    nodes = 0

    def explore(i, used_time, score):
        nonlocal best_value, best_indices, nodes
        nodes += 1
        if score > best_value:
            best_value = score
            best_indices = list(chosen)
        if i == n:
            return
        w = modules[i].time
        if used_time + w <= budget:
            chosen.append(i)
            explore(i + 1, used_time + w, score + modules[i].score)
            chosen.pop()
        explore(i + 1, used_time, score)

    explore(0, 0, 0)
    return Result("DFS", best_value, best_indices, nodes)
