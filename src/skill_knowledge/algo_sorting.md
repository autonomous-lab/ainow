---
name: sorting_choice
triggers: ["sort", "sorted", "order by", "sort stable", "custom sort key", "radix sort", "counting sort", "top-k"]
max_chars: 1500
---

## Picking the right sort

- **Default**: `sorted(iterable, key=…)` or `list.sort()` in Python — Timsort, stable, O(n log n). Don't reinvent it.
- **Custom key beats custom compare**: `sorted(items, key=lambda x: (x.priority, -x.age))` is both faster and clearer than `functools.cmp_to_key`.
- **Top-k without full sort**: `heapq.nlargest(k, it, key=…)` / `nsmallest` — O(n log k) instead of O(n log n). Big win when k << n.
- **Partial sort**: `heapq.heapify(xs)` then pop k times. Useful when you only consume the smallest items lazily.
- **Bucket/counting sort**: when the key space is bounded and small (int 0–1000, chars, etc.), counting sort is O(n + k) and easy. For floats, use `numpy.argsort` if the array is numeric.
- **Stability matters** when you sort by multiple keys: do the *least significant* key first, then the next, because each pass preserves the previous ordering.
- **Already-sorted input**: don't re-sort. `sorted(sorted_list)` is O(n) due to Timsort's run detection, but it's still a wasted allocation.
