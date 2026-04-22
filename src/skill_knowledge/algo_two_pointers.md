---
name: two_pointers
triggers: ["two pointers", "sliding window", "fast slow pointer", "partition array", "subarray sum"]
max_chars: 1200
---

## Two pointers / sliding window

- **When it applies**: a contiguous subarray / substring problem with a monotonic property (sum ≥ target, distinct count ≤ k, …). If extending the window can only make the predicate more true, and shrinking can only make it more false, it's a sliding-window problem.
- **Shape**:
  ```
  left = 0
  for right in range(n):
      add a[right]
      while window invalid:
          remove a[left]; left += 1
      # window [left..right] is now valid — update answer
  ```
- **Fast/slow pointer**: cycle detection in a linked list or array — move fast by 2 and slow by 1, they meet iff there's a cycle.
- **Opposite-ends pointers**: sum / partition problems on a *sorted* array. Start `lo=0`, `hi=n-1`, move inward based on the sum/condition.
- **Watch integer-vs-character indexing** in string windows with Unicode: a naive `s[i:j]` on a string with multi-byte characters is fine in Python but a trap in Rust/Go.
