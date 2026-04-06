# Git & CI Process

## Repository

- **Repo**: https://github.com/autonomous-lab/ainow
- **Author**: jbenguira <jbenguira@users.noreply.github.com>
- **PAT**: stored in `.env` as `GITHUB_PAT` (scopes: repo, workflow, actions)

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable Python version (FastAPI + browser UI) |
| `rust` | Rust/Tauri port (Axum + Tauri desktop app) |

## Push Commands

PAT is in `.env`. To push:

```bash
# Load PAT from .env
source <(grep GITHUB_PAT .env)

# Push to main
git push https://autonomous-lab:$GITHUB_PAT@github.com/autonomous-lab/ainow.git main

# Push to rust
git push https://autonomous-lab:$GITHUB_PAT@github.com/autonomous-lab/ainow.git rust

# Clean PAT from remote after push
git remote set-url origin https://github.com/autonomous-lab/ainow.git
```

## CI / GitHub Actions

- **Workflow**: `.github/workflows/build.yml` (on `rust` branch only)
- **Triggers**: push or PR to `rust` branch
- **Builds**: Windows (MSVC), Linux (Ubuntu), macOS (Apple Silicon)
- **Steps**: install Rust + Node.js, `cargo check`, `tauri build`, upload artifacts
- **Artifacts**: downloadable from Actions tab after successful build

### Check CI status

```bash
source <(grep GITHUB_PAT .env)
curl -s -H "Authorization: token $GITHUB_PAT" \
  https://api.github.com/repos/autonomous-lab/ainow/actions/runs \
  | python -c "import sys,json; [print(f\"{r['status']} {r.get('conclusion','')} {r['name']}\") for r in json.load(sys.stdin).get('workflow_runs',[])[:5]]"
```

### Read CI logs for a job

```bash
# Get job IDs for a run
curl -s -H "Authorization: token $GITHUB_PAT" \
  https://api.github.com/repos/autonomous-lab/ainow/actions/runs/<RUN_ID>/jobs \
  | python -c "import sys,json; [print(f\"{j['id']} {j['name']}: {j.get('conclusion','')}\") for j in json.load(sys.stdin)['jobs']]"

# Get logs for a job
curl -s -H "Authorization: token $GITHUB_PAT" -L \
  https://api.github.com/repos/autonomous-lab/ainow/actions/jobs/<JOB_ID>/logs
```

## Commit Conventions

- Author: `jbenguira <jbenguira@users.noreply.github.com>`
- No co-author credits
- Commit messages: imperative mood, short first line, optional body

## Git Config (per-repo)

```bash
git config user.name "jbenguira"
git config user.email "jbenguira@users.noreply.github.com"
```

## WDAC Note

This Windows machine has WDAC (Windows Defender Application Control) policies that block unsigned binaries (cargo build scripts). Rust compilation must be done via GitHub Actions CI, not locally. Do NOT delete WDAC `.cip` policy files — this will brick Windows boot.
