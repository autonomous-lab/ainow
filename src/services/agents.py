"""
Agent store: per-agent CLAUDE.md, sessions, and memory.

Layout (relative to the directory the server was launched from):
  <cwd>/agents/<name>/
    CLAUDE.md           # System prompt for this agent
    meta.json           # {display_name, created_at}
    sessions/*.json     # Conversation history
    memory/*.md         # Persistent memories

  <cwd>/agents/.active  # Plain text file containing the active agent name

A "default" agent is auto-created on first use. Legacy locations are migrated
once: ~/.ainow/agents/* or ~/.ainow/sessions / ~/.ainow/memory if present.
"""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..log import get_logger

logger = get_logger("ainow.agents")

AGENTS_ROOT = Path.cwd() / "agents"
ACTIVE_FILE = AGENTS_ROOT / ".active"
DEFAULT_AGENT = "default"

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_name(name: str) -> None:
    if not name or not _NAME_RE.match(name):
        raise ValueError(f"Invalid agent name '{name}' (use alphanumerics, underscore, dash)")


def agent_dir(name: str) -> Path:
    _validate_name(name)
    return AGENTS_ROOT / name


def sessions_dir(name: str) -> Path:
    return agent_dir(name) / "sessions"


def claude_md_path(name: str) -> Path:
    return agent_dir(name) / "CLAUDE.md"


def meta_path(name: str) -> Path:
    return agent_dir(name) / "meta.json"


def exists(name: str) -> bool:
    try:
        return agent_dir(name).is_dir()
    except ValueError:
        return False


def _write_meta(name: str) -> None:
    meta = {
        "display_name": name,
        "created_at": datetime.now().isoformat(),
        "mcp_servers": {},
    }
    meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(name: str) -> dict:
    """Read an agent's meta.json (returns {} if missing/unreadable)."""
    p = meta_path(name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_meta(name: str, meta: dict) -> None:
    """Overwrite an agent's meta.json with a merged dict."""
    if not exists(name):
        raise ValueError(f"Agent '{name}' does not exist")
    meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_mcp_servers(name: str) -> dict:
    """Return the agent's configured MCP servers (name -> {command, args, env})."""
    return read_meta(name).get("mcp_servers", {}) or {}


def write_mcp_servers(name: str, servers: dict) -> None:
    """Update only the mcp_servers field of an agent's meta.json."""
    meta = read_meta(name)
    meta["mcp_servers"] = servers
    write_meta(name, meta)


def read_preferences(name: str) -> dict:
    """Return per-agent UI preferences (lang, voice) from meta.json."""
    return read_meta(name).get("preferences", {}) or {}


def update_preferences(name: str, patch: dict) -> dict:
    """Merge a patch into the agent's preferences."""
    meta = read_meta(name)
    prefs = meta.get("preferences", {}) or {}
    prefs.update({k: v for k, v in patch.items() if v is not None})
    meta["preferences"] = prefs
    write_meta(name, meta)
    return prefs


# ── Scheduled tasks per agent ─────────────────────────────────────────

def scheduled_tasks_path(name: str) -> Path:
    return agent_dir(name) / "scheduled_tasks.json"


def read_scheduled_tasks(name: str) -> List[dict]:
    """Return the agent's scheduled tasks list (empty list if missing)."""
    p = scheduled_tasks_path(name)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        return data.get("tasks", []) or []
    return data or []


def write_scheduled_tasks(name: str, tasks: List[dict]) -> None:
    if not exists(name):
        raise ValueError(f"Agent '{name}' does not exist")
    scheduled_tasks_path(name).write_text(
        json.dumps({"tasks": tasks}, indent=2),
        encoding="utf-8",
    )


def add_scheduled_task(name: str, task: dict) -> dict:
    """Append a task. Returns the stored task with id/timestamps filled in."""
    tasks = read_scheduled_tasks(name)
    if "id" not in task:
        task["id"] = f"task_{int(datetime.now().timestamp())}_{os.urandom(3).hex()}"
    task.setdefault("created_at", datetime.now().isoformat())
    task.setdefault("enabled", True)
    task.setdefault("mode", "new_session")  # or "inject"
    task.setdefault("run_count", 0)
    task.setdefault("last_run", None)
    task.setdefault("last_status", None)
    tasks.append(task)
    write_scheduled_tasks(name, tasks)
    return task


def update_scheduled_task(name: str, task_id: str, patch: dict) -> Optional[dict]:
    tasks = read_scheduled_tasks(name)
    for i, t in enumerate(tasks):
        if t.get("id") == task_id:
            tasks[i] = {**t, **patch}
            write_scheduled_tasks(name, tasks)
            return tasks[i]
    return None


def delete_scheduled_task(name: str, task_id: str) -> bool:
    tasks = read_scheduled_tasks(name)
    new_tasks = [t for t in tasks if t.get("id") != task_id]
    if len(new_tasks) == len(tasks):
        return False
    write_scheduled_tasks(name, new_tasks)
    return True


def create(name: str, claude_md: str = "") -> None:
    _validate_name(name)
    if exists(name):
        raise ValueError(f"Agent '{name}' already exists")
    d = agent_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    sessions_dir(name).mkdir(exist_ok=True)
    claude_md_path(name).write_text(claude_md, encoding="utf-8")
    _write_meta(name)
    logger.info(f"Created agent '{name}'")


def delete(name: str) -> None:
    _validate_name(name)
    if name == DEFAULT_AGENT:
        raise ValueError("Cannot delete the default agent")
    if not exists(name):
        raise ValueError(f"Agent '{name}' does not exist")
    shutil.rmtree(agent_dir(name))
    if get_active() == name:
        set_active(DEFAULT_AGENT)
    logger.info(f"Deleted agent '{name}'")


def list_agents() -> List[dict]:
    """Return [{name, active, created_at}], sorted with default first."""
    if not AGENTS_ROOT.exists():
        return []
    active = get_active()
    out = []
    for child in sorted(AGENTS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        try:
            _validate_name(name)
        except ValueError:
            continue
        created = ""
        try:
            meta = json.loads(meta_path(name).read_text(encoding="utf-8"))
            created = meta.get("created_at", "")
        except Exception:
            pass
        out.append({"name": name, "active": name == active, "created_at": created})
    out.sort(key=lambda a: (a["name"] != DEFAULT_AGENT, a["name"]))
    return out


def read_claude_md(name: str) -> str:
    p = claude_md_path(name)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def write_claude_md(name: str, content: str) -> None:
    if not exists(name):
        raise ValueError(f"Agent '{name}' does not exist")
    claude_md_path(name).write_text(content, encoding="utf-8")


def get_active() -> str:
    """Return active agent name, falling back to default. Side-effect free."""
    if ACTIVE_FILE.exists():
        try:
            name = ACTIVE_FILE.read_text(encoding="utf-8").strip()
            if name and exists(name):
                return name
        except Exception:
            pass
    return DEFAULT_AGENT


def set_active(name: str) -> None:
    _validate_name(name)
    if not exists(name):
        raise ValueError(f"Agent '{name}' does not exist")
    AGENTS_ROOT.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.write_text(name, encoding="utf-8")
    logger.info(f"Active agent: {name}")


def ensure_default() -> None:
    """Create default agent if missing. Migrate legacy paths once."""
    # Legacy migration: ~/.ainow/agents -> <cwd>/agents (full move)
    legacy_agents_root = Path(os.path.expanduser("~")) / ".ainow" / "agents"
    if legacy_agents_root.is_dir() and not AGENTS_ROOT.exists():
        try:
            shutil.move(str(legacy_agents_root), str(AGENTS_ROOT))
            logger.info(f"Migrated agents from {legacy_agents_root} to {AGENTS_ROOT}")
        except Exception as e:
            logger.error(f"Agent migration failed: {e}")

    AGENTS_ROOT.mkdir(parents=True, exist_ok=True)

    if not exists(DEFAULT_AGENT):
        seed = ""
        # Seed CLAUDE.md from cwd if present (one-time)
        cwd_claude = Path.cwd() / "CLAUDE.md"
        if cwd_claude.exists():
            try:
                seed = cwd_claude.read_text(encoding="utf-8")
                logger.info(f"Seeding default agent from {cwd_claude}")
            except Exception:
                pass
        create(DEFAULT_AGENT, seed)

    # Legacy migration: ~/.ainow/sessions -> agents/default/sessions
    legacy_sessions = Path(os.path.expanduser("~")) / ".ainow" / "sessions"
    if legacy_sessions.is_dir():
        target = sessions_dir(DEFAULT_AGENT)
        target.mkdir(exist_ok=True)
        moved = 0
        for f in legacy_sessions.iterdir():
            if f.is_file():
                dest = target / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
                    moved += 1
        try:
            legacy_sessions.rmdir()
        except OSError:
            pass
        if moved:
            logger.info(f"Migrated {moved} sessions to agents/default/sessions")

    if not ACTIVE_FILE.exists():
        set_active(DEFAULT_AGENT)
