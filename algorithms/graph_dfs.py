def dfs_paths(successors, times, scores, source, budget):
    best_score, best_path, generated = -1, [], 0
    path = []

    def visit(node, used_time, score):
        nonlocal best_score, best_path, generated
        path.append(node)
        used_time += times[node]
        score += scores[node]
        generated += 1
        if score > best_score:
            best_score, best_path = score, list(path)
        for nxt in successors.get(node, []):
            if nxt not in path and used_time + times[nxt] <= budget:
                visit(nxt, used_time, score)
        path.pop()

    if times[source] <= budget:
        visit(source, 0, 0)
    return best_score, best_path, generated


def complete_dag(n):
    return {i: list(range(i + 1, n)) for i in range(n)}
