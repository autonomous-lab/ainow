---
name: git_workflow
triggers: ["git commit", "git diff", "git log", "git push", "git status", "commit message", "branch"]
max_chars: 1800
---

## Git workflow — safe defaults

- **Inspect before you commit**: run `git status`, `git diff` (or `git diff --staged`), and `git log --oneline -5` — in parallel — to know exactly what you're about to commit and the repo's existing commit style.
- **Stage explicitly**: `git add path1 path2` rather than `git add -A`. The blanket stage accidentally includes `.env`, credentials, or large binaries.
- **Never amend pushed commits** without checking with the user first. Amending rewrites history and force-push is required to propagate — both are high-blast-radius.
- **Never push directly to `main`/`master`** if a PR workflow exists. Create a branch, push that, open a PR.
- **Destructive commands** (`git reset --hard`, `git checkout --`, `git clean -f`, `git branch -D`): confirm with the user before running. They silently destroy uncommitted work.
- **Commit messages**: write them in the repo's existing style (short imperative summary, blank line, optional body). Check recent `git log --oneline` first.
- When hooks fail on commit, the commit did NOT happen. Fix the underlying issue and create a **new** commit — don't `--amend` (that modifies the *previous* commit, not the failed one).
- If there are no changes to commit, don't create an empty commit. Report "nothing to commit" and stop.
