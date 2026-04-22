---
name: hash_vs_tree
triggers: ["hashmap", "hash table", "dict", "ordered map", "sorted set", "HashMap vs TreeMap"]
max_chars: 1100
---

## Hash map vs ordered tree — not interchangeable

- **Hash (`dict`, `set`)**: O(1) average insert / lookup / delete. No ordering. Use for membership and counting.
- **Ordered (`sortedcontainers.SortedDict`, `bisect` on a list, Java's `TreeMap`)**: O(log n) insert / lookup, plus range queries (`irange`) and nearest-neighbour (`bisect`). Use when you need:
  - "smallest key ≥ x"
  - iterate in sorted order
  - range-delete
- **Counter quirks**: `collections.Counter(it)` gives a `dict`-subclass with vector math (`+`, `-`, `&`, `|`). `.most_common(k)` is the one-liner for top-k frequencies.
- **Don't mutate a dict while iterating it**: snapshot keys first with `list(d.keys())` or use `list(d.items())`.
- **Frozen key types**: dict keys must be hashable. A list key fails at runtime; convert to `tuple` or use a canonical string form.
