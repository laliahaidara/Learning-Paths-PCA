"""DFS over a prerequisite graph: the paper's RUTICO search, as published.

Nodes are modules and edges say "may come next". The search enumerates every
simple path that starts at ``source`` and fits in the time budget. On a
complete DAG with n nodes there are 2^(n-1) such paths, which is the
exponential cost the knapsack DP avoids (see the ``graph`` experiment).
"""


def dfs_paths(successors, times, scores, source, budget):
    """Best-scoring path from ``source`` within ``budget``.

    ``successors`` maps a node to the list of nodes that may follow it.
    Returns ``(best score, best path, number of paths generated)``.
    """
    best_score, best_path, generated = -1, [], 0
    path = []

    def visit(node, used_time, score):
        nonlocal best_score, best_path, generated
        path.append(node)
        used_time += times[node]
        score += scores[node]
        generated += 1  # every prefix of a path is itself a path
        if score > best_score:
            best_score, best_path = score, list(path)
        for nxt in successors.get(node, []):
            if nxt not in path and used_time + times[nxt] <= budget:
                visit(nxt, used_time, score)
        path.pop()  # backtrack

    if times[source] <= budget:
        visit(source, 0, 0)
    return best_score, best_path, generated


def complete_dag(n):
    """Complete DAG on nodes 0..n-1: every node points to all later nodes."""
    return {i: list(range(i + 1, n)) for i in range(n)}
