---
name: debugging
triggers: ["bug", "crash", "error", "traceback", "exception", "stack trace", "why does", "not working", "broken"]
max_chars: 1500
---

## Debugging — diagnose before you fix

- **Read the full error**, including the traceback, before proposing a fix. The line number and the exception type are usually enough to localize the cause; don't skip the 5 lines of "too obvious" context.
- **Reproduce it first**: can you trigger the failure with a minimal command? If not, you don't know enough yet to fix it.
- **Check assumptions, don't mask them**: if you're tempted to wrap something in `try/except` to make the error go away, stop and ask *why* it's throwing. Silent-catch is almost always wrong.
- **Compare against a known-good state**: if it worked yesterday, `git log -p` or `git bisect` will find the break faster than guessing.
- **Don't blind-retry the same action** after a failure. Change exactly one thing between attempts so you can attribute the outcome.
- When reporting to the user, lead with the **root cause** in one sentence, then the fix. Don't narrate your search process.
- If you can't verify the fix end-to-end (no test, no runnable repro), say so explicitly. Don't claim "should work" — that's future-you's problem.
