---
name: dynamic_programming
triggers: ["dynamic programming", "memoize", "memoization", "overlapping subproblems", "DP table", "bottom-up", "top-down"]
max_chars: 1600
---

## Dynamic programming — choose representation before coding

- **State first, recurrence second**: before writing code, name the state tuple (e.g. `(i, j, remaining_budget)`) and write the recurrence as one equation. If you can't write the recurrence in one line, you don't understand the problem yet.
- **Top-down (memo) is easier to get right**: write the naive recursion, then decorate with `@functools.lru_cache(maxsize=None)`. No table-allocation bugs, no dimension-ordering bugs.
- **Bottom-up (tabulation) is faster and uses less stack**: switch to it once the recurrence is validated end-to-end, or when recursion depth would exceed Python's 1000-frame default.
- **Dimension reduction**: if the recurrence only reaches back one row (e.g. `dp[i][j] = dp[i-1][j] + dp[i-1][j-1]`), collapse to two 1-D arrays (`prev`, `curr`) and swap each iteration. Memory goes from O(n·m) to O(m).
- **Print the table** while debugging small inputs. Most DP bugs reveal themselves as one mispopulated cell.
- **Watch the base case**: `dp[0]` or `dp[0][0]` almost always needs an explicit value, not an uninitialized zero. `INF` for minimization, `-INF` for maximization, `0` for counting.
- **Problem reductions to know**: Knapsack, LCS, edit distance, coin change, LIS (with patience sort trick: O(n log n) instead of O(n²)).
