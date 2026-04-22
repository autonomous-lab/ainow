---
name: binary_search
triggers: ["binary search", "bisect", "lower_bound", "upper_bound", "logarithmic search"]
max_chars: 1500
---

## Binary search — decision-framing works better than index-chasing

- **Always use half-open intervals** `[lo, hi)`. Let `lo = 0`, `hi = n` (not `n-1`). Loop while `lo < hi`, midpoint `m = (lo + hi) // 2`. After the loop, `lo == hi` and that's the insertion point.
- **Python stdlib**: `bisect.bisect_left(a, x)` returns the leftmost insertion index; `bisect_right` the rightmost. Use these instead of hand-rolled loops unless you need a custom predicate.
- **Binary search over answers**: if the feasibility function `f(x)` is monotonic (True for small x, False for large x — or vice versa), you can binary-search on `x` itself even when the input is continuous. The loop is the same; the middle check calls `f(m)`.
- **Off-by-one test**: always hand-run on arrays of size 0, 1, and 2 with a target smaller than / equal to / larger than all elements.
- Overflow isn't a Python concern but matters in C/Java/Rust: use `lo + (hi - lo) // 2`, not `(lo + hi) // 2`.
