---
name: dfs_vs_bfs
triggers: ["traversal", "graph search", "BFS", "DFS", "shortest path", "connected components", "topological", "cycle detection"]
max_chars: 1400
---

## DFS vs BFS — pick by the problem shape

- **Shortest path in an unweighted graph** → BFS. Use a `collections.deque`; append on the right, popleft from the left. `dict` for `dist`, not a matrix unless the graph is dense.
- **Shortest path with non-negative weights** → Dijkstra (`heapq.heappush` / `heappop`). Don't add to the heap with `visited` — mark on pop.
- **Any path / reachability / topological sort / cycle detection** → DFS. Iterative DFS with an explicit stack avoids Python's recursion-depth limit on deep graphs.
- **Connected components**: union-find is simpler and faster than DFS when components shift over time (dynamic connectivity).
- **Grid DFS/BFS**: neighbours = `[(-1,0), (1,0), (0,-1), (0,1)]`. Always guard `0 <= r < R and 0 <= c < C` before indexing.
- **Mark on push, not on pop**, or you enqueue duplicates. The visited set grows before dequeue.
- **Bidirectional BFS** halves the frontier size when you know both start and end — pick it when the branching factor is high.
