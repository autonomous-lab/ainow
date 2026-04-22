"""AINow CLI — use AINow as a headless local coding / chat agent.

A Rich-powered TUI with startup banner, live context status line, formatted
tool-use flow, and slash commands. Inspired by little-coder's presentation.

Examples:
    python -m src.cli "list the biggest python files under src/"
    python -m src.cli -i                       # REPL
    python -m src.cli -a donald-trump -m 9b
    python -m src.cli --yolo "refactor foo.py"  # auto-approve tool calls

Slash commands inside the REPL:
    /help              show commands
    /model [alias]     switch or show model
    /agent [name]      switch or show agent
    /clear             reset conversation history
    /compact           force context compaction now
    /context           detailed context breakdown
    /quit | /exit      leave
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# Windows consoles default to cp1252 / cp437, which silently mangles UTF-8 from
# the LLM: multi-byte codepoints (emoji, curly quotes, em-dash) get split
# across write() calls and decoded wrong, producing broken flag glyphs and
# spurious line breaks. Force UTF-8 on both streams before anything prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# The Windows asyncio proactor spews ResourceWarning / "Event loop is closed"
# tracebacks on interpreter exit when a subprocess transport is still in the
# finalizer queue. These are cosmetic; silence them so a Ctrl-C during `!cmd`
# doesn't produce a wall of scary red output.
if sys.platform == "win32":
    warnings.filterwarnings("ignore", category=ResourceWarning, module=r"asyncio\..*")
    try:
        import logging as _logging
        _logging.getLogger("asyncio").setLevel(_logging.ERROR)
    except Exception:
        pass

# Load .env before any service imports so LLAMA_SERVER_EXE / MODELS_DIR /
# MODEL_<ALIAS> / KV_CACHE_TYPE / AINOW_* are picked up the same way main.py
# does. Without this the CLI would default to PATH-resolved `llama-server`.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Readline gives us ↑/↓ history, Ctrl+A/E/U/K/W line editing, reverse-search
# in the REPL. On POSIX it's stdlib; on Windows an optional `pyreadline3`
# package fills the same shape. input() picks it up automatically once imported.
try:
    import readline  # noqa: F401  (POSIX)
    _HAS_READLINE = True
except ImportError:
    try:
        import pyreadline3 as readline  # noqa: F401  (Windows opt-in)
        _HAS_READLINE = True
    except ImportError:
        _HAS_READLINE = False

_HISTFILE = os.path.join(os.path.expanduser("~"), ".ainow_cli_history")
if _HAS_READLINE:
    try:
        readline.read_history_file(_HISTFILE)
    except (FileNotFoundError, OSError):
        pass
    try:
        readline.set_history_length(1000)
    except Exception:
        pass
    import atexit
    def _save_history():
        try:
            readline.write_history_file(_HISTFILE)
        except OSError:
            pass
    atexit.register(_save_history)

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.align import Align


# Single shared console — stderr by default so piping `prompt` still works
# cleanly. Token stream goes to stdout for piping into other tools.
console = Console(file=sys.stderr, highlight=False, soft_wrap=True)
stdout_console = Console(file=sys.stdout, highlight=False, soft_wrap=True)

VERSION = "v1.1"
STREAMED_TOKENS_IN_TURN = 0


# ──────────────────────────────────────────────────────────────────────────
# Presentation helpers
# ──────────────────────────────────────────────────────────────────────────

def _banner(model_label: str, backend: str, permissions: str, ctx: int, agent: str, minimal: bool = False) -> None:
    """little-coder-style startup banner."""
    if minimal:
        perm_color = "yellow" if permissions == "yolo" else "green"
        console.print(Text.from_markup(
            f"[bold magenta]AINow[/bold magenta] [dim]{VERSION}[/dim]  ·  "
            f"[cyan]{model_label}[/cyan] [dim]({backend})[/dim]  ·  "
            f"agent [cyan]{agent}[/cyan]  ·  "
            f"[{perm_color}]{permissions}[/]  ·  "
            f"ctx [cyan]{_fmt_ctx(ctx) if ctx else 'unknown'}[/cyan]  ·  "
            f"[dim]/help[/dim]"
        ))
        return
    body = Group(
        Text.from_markup(f"[bold magenta]AINow[/bold magenta]  ·  [dim]{VERSION}[/dim]"),
        Text.from_markup("[italic]Local-first AI agent framework — chat, code, voice, vision[/italic]"),
        Text(""),
        Text.from_markup(f"[dim]model:       [/dim][cyan]{model_label}[/cyan]  [dim]({backend})[/dim]"),
        Text.from_markup(f"[dim]agent:       [/dim][cyan]{agent}[/cyan]"),
        Text.from_markup(f"[dim]permissions: [/dim][{'yellow' if permissions == 'yolo' else 'green'}]{permissions}[/{'yellow' if permissions == 'yolo' else 'green'}]"),
        Text.from_markup(f"[dim]context:     [/dim][cyan]{_fmt_ctx(ctx) if ctx else 'unknown'}[/cyan]"),
        Text(""),
        Text.from_markup("[dim]/help for commands  ·  /model to switch  ·  /quit to exit[/dim]"),
    )
    console.print(Panel(body, border_style="magenta", padding=(0, 2)))


# Spinner phrases shown while waiting for the first LLM token.
_SPINNER_PHRASES = [
    "AINow is thinking…",
    "consulting the local oracle…",
    "warming up the KV cache…",
    "asking the silicon nicely…",
    "decoding…",
    "crunching tokens…",
    "listening to the GPU hum…",
]


def _random_spinner() -> str:
    import random
    return random.choice(_SPINNER_PHRASES)


def _fmt_ctx(n: int) -> str:
    if n >= 1_048_576:
        return f"{n // 1024 // 1024}M"
    if n >= 1024:
        return f"{n // 1024}K"
    return str(n)


def _status_line(used: int, ctx_max: int, model_label: str) -> None:
    """Context status line shown before each REPL prompt."""
    if not ctx_max:
        console.print(Text.from_markup(f"[dim]context: unknown[/dim]  [dim]·[/dim]  [cyan]{model_label}[/cyan]"))
        return
    pct = min(100, int(used / ctx_max * 100)) if ctx_max else 0
    # Zone colouring: 0-70 green, 70-85 yellow, 85+ red
    if pct < 70:
        color, zone = "green", "green"
    elif pct < 85:
        color, zone = "yellow", "yellow — context getting tight"
    else:
        color, zone = "red", "red — /compact or /clear"
    used_fmt = _fmt_ctx(used)
    max_fmt = _fmt_ctx(ctx_max)
    # Rough msgs-remaining estimate (assumes ~1500 tok per user+assistant pair)
    remaining_tok = max(0, ctx_max - used)
    msgs_remaining = max(0, remaining_tok // 1500)
    msgs_str = f"~{msgs_remaining} msgs" if msgs_remaining > 0 else "limit reached"
    msgs_color = color
    line = Text.assemble(
        ("context: ", "dim"),
        (f"{used_fmt}/{max_fmt}", color),
        (f" ({pct}%)", color),
        ("  ·  ", "dim"),
        (msgs_str + " until recommended new session", msgs_color),
        ("  ·  ", "dim"),
        ("model: ", "dim"),
        (model_label, "cyan"),
    )
    console.print(line)
    if pct >= 85:
        console.print(Text.from_markup(f"[red]zone: {zone}[/red]"))


def _prompt() -> Text:
    return Text.from_markup("[bold magenta]›[/bold magenta] ")


# ──────────────────────────────────────────────────────────────────────────
# Tool-use flow formatting
# ──────────────────────────────────────────────────────────────────────────

def _short_arg(args) -> str:
    """Pick the most identifying arg value for a one-line label."""
    if not isinstance(args, dict):
        try:
            import json
            args = json.loads(args) if isinstance(args, str) else {}
        except Exception:
            args = {}
    if not isinstance(args, dict):
        return ""
    for k in ("path", "file", "file_path", "pattern", "query", "q",
              "command", "url", "name", "text", "message"):
        v = args.get(k)
        if isinstance(v, str) and v:
            if len(v) > 80:
                v = v[:77] + "…"
            return v
        if isinstance(v, (int, float)) and v:
            return str(v)
    # Fallback: first non-empty string-ish value
    for v in args.values():
        if isinstance(v, str) and v:
            return v[:80]
    return ""


def _tool_arrow(name: str, args) -> None:
    label = _short_arg(args)
    line = Text.assemble(
        ("→ ", "bold cyan"),
        (f"{name:<8}", "cyan"),
        ("  ", ""),
        (label, "dim"),
    )
    console.print(line)


def _tool_check(name: str, result: str) -> None:
    # Heuristic result summary — first line, or size hint for large results.
    result = result or ""
    n_lines = result.count("\n") + (1 if result.strip() else 0)
    n_chars = len(result)
    is_error = result.startswith(("Error", "error", "[ERROR")) or "[exit code:" in result[:200]
    mark = "✗" if is_error else "✓"
    color = "red" if is_error else "green"
    if is_error:
        first = result.strip().splitlines()[0] if result.strip() else "(error)"
        if len(first) > 120:
            first = first[:117] + "…"
        detail = first
    elif n_lines > 1:
        detail = f"{n_lines} lines ({n_chars} chars)"
    elif n_chars > 0:
        snippet = result.strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "…"
        detail = snippet
    else:
        detail = "(no output)"
    line = Text.assemble(
        (f"{mark} ", f"bold {color}"),
        ("→ ", f"{color}"),
        (f"{name:<8}", color),
        ("  ", ""),
        (detail, "dim"),
    )
    console.print(line)


# ──────────────────────────────────────────────────────────────────────────
# Async LLM plumbing
# ──────────────────────────────────────────────────────────────────────────

async def _print_token(token: str) -> None:
    global STREAMED_TOKENS_IN_TURN
    STREAMED_TOKENS_IN_TURN += 1
    # Stream raw to stdout so output is pipeable and mixes cleanly with Rich
    # status lines (which go to stderr).
    sys.stdout.write(token)
    sys.stdout.flush()


async def _on_tool_call(name: str, args) -> None:
    # Ensure token stream has a newline before the tool arrow
    if STREAMED_TOKENS_IN_TURN > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
    _tool_arrow(name, args)


_CURRENT_STATE: Optional["CLIState"] = None  # set in main() so callbacks can reach flags


async def _on_tool_result(tool_call_id: str, name: str, result: str) -> None:
    _tool_check(name, result)
    # Expanded tool-output mode (toggle via Ctrl+O): also print the full body,
    # truncated to keep the terminal usable.
    if _CURRENT_STATE is not None and _CURRENT_STATE.show_tool_output and result:
        body = result if len(result) <= 2000 else result[:2000] + "\n… (truncated)"
        console.print(Panel(Text(body), border_style="dim", padding=(0, 1)))


async def _on_thinking(text: str, duration: float, done: bool) -> None:
    """Stream reasoning_content when Ctrl+T toggle is on, else stay silent."""
    if _CURRENT_STATE is None or not _CURRENT_STATE.show_thinking:
        return
    if done:
        sys.stderr.write(f"\n[thinking done in {duration:.1f}s]\n")
        sys.stderr.flush()
        return
    if text:
        sys.stderr.write(f"\033[2m{text}\033[0m")  # dim ANSI
        sys.stderr.flush()


def _prompt_confirm(name: str, args) -> bool:
    """Synchronous y/N prompt for dangerous tools."""
    console.print(
        Text.assemble(
            ("? ", "bold yellow"),
            ("confirm ", "yellow"),
            (f"{name}", "cyan"),
            (" ", ""),
            (_short_arg(args), "dim"),
            (" — run? [y/N] ", "dim"),
        ),
        end="",
    )
    try:
        answer = sys.stdin.readline()
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


# ──────────────────────────────────────────────────────────────────────────
# State container so slash commands can mutate it
# ──────────────────────────────────────────────────────────────────────────

class CLIState:
    def __init__(self, yolo: bool):
        self.yolo = yolo
        self.llm = None          # LLMService — created on first turn
        self.model_alias: str = "9b"
        self.agent_name: str = ""
        self.verbose: bool = False  # toggle via /verbose
        # TUI toggles (Ctrl+O, Ctrl+T)
        self.show_tool_output: bool = False  # when True, print full tool results
        self.show_thinking: bool = False  # when True, stream reasoning_content live
        # Cached context usage — refreshed after each turn, read by the toolbar
        # on every keystroke. Computing context_used from scratch would re-run
        # _build_system_prompt (which shells out to git) on every key event.
        self.cached_context_used: int = 0
        self.cached_context_max: int = 0
        # Monotonic deadline after which a pending Shift+Tab (thinking toggle)
        # is no longer armed. 0 = not armed. First press arms, second press
        # within the window actually toggles (reloads llama-server).
        self.thinking_toggle_armed_until: float = 0.0

    async def confirm(self, name: str, args) -> bool:
        if self.yolo:
            console.print(
                Text.assemble(("[auto] ", "dim yellow"), (f"approved {name}", "dim"))
            )
            return True
        return _prompt_confirm(name, args)

    def _ensure_llm(self):
        if self.llm is not None:
            return
        from .services.llm import LLMService
        from .services.tools import get_tool_schemas
        self.llm = LLMService(
            on_token=_print_token,
            on_done=self._noop_done,
            tools=get_tool_schemas(),
            on_tool_call=_on_tool_call,
            on_tool_result=_on_tool_result,
            on_tool_confirm=self.confirm,
            on_thinking=_on_thinking,
        )

    async def _noop_done(self) -> None:
        pass

    def context_stats(self) -> tuple[int, int]:
        """Return (used, max) using the cached values.

        The toolbar calls this on every keystroke — recomputing context_used
        for real would re-invoke _build_system_prompt (which shells out to git),
        turning every character of typing into a several-ms spend. The cache
        is refreshed via refresh_context_stats() after each turn.
        """
        return self.cached_context_used, self.cached_context_max

    def refresh_context_stats(self) -> None:
        """Recompute + cache context usage. Called between turns, not per-key."""
        if self.llm is None:
            self.cached_context_used = 0
            try:
                from .services.model_manager import model_manager
                self.cached_context_max = model_manager.get_context_size()
            except Exception:
                self.cached_context_max = 0
            return
        try:
            self.cached_context_used = self.llm.context_used
            self.cached_context_max = self.llm.context_max
        except Exception:
            pass

    def backend_label(self) -> str:
        from .services.model_manager import MODELS, resolve_model_id
        try:
            mid = resolve_model_id(self.model_alias)
        except Exception:
            return "unknown"
        cfg = MODELS.get(mid, {})
        return "cloud" if cfg.get("online") else "llama.cpp"

    def model_label(self) -> str:
        from .services.model_manager import MODELS, resolve_model_id
        try:
            mid = resolve_model_id(self.model_alias)
            return MODELS[mid].get("name", mid)
        except Exception:
            return self.model_alias


# ──────────────────────────────────────────────────────────────────────────
# Slash commands
# ──────────────────────────────────────────────────────────────────────────

async def _handle_slash(state: "CLIState", line: str) -> bool:
    """Return True if the line was a slash command (REPL should skip LLM call)."""
    if not line.startswith("/"):
        return False
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit", "q"):
        raise SystemExit(0)

    if cmd == "help":
        console.print(Panel(
            Text.from_markup(
                "[bold]Session[/bold]\n"
                "  [cyan]/help[/cyan]                show this message\n"
                "  [cyan]/quit[/cyan] · [cyan]/exit[/cyan]         leave\n"
                "\n"
                "[bold]Model / agent[/bold]\n"
                "  [cyan]/model [alias][/cyan]       switch or show model\n"
                "  [cyan]/agent[/cyan]               list all agents with status\n"
                "  [cyan]/agent <name>[/cyan]        switch to an agent\n"
                "  [cyan]/agent new <name>[/cyan]    create a new agent\n"
                "  [cyan]/agent delete <name>[/cyan] delete an agent\n"
                "  [cyan]/agent edit [name][/cyan]   open CLAUDE.md in $EDITOR\n"
                "  [cyan]/agent info [name][/cyan]   full agent detail\n"
                "  [cyan]/thinking[/cyan]            toggle reasoning mode (reloads llama-server)\n"
                "  [cyan]/permissions [mode][/cyan]  switch [magenta]yolo[/magenta] / [magenta]confirm[/magenta]\n"
                "\n"
                "[bold]Context[/bold]\n"
                "  [cyan]/context[/cyan]             detailed context breakdown\n"
                "  [cyan]/history[/cyan]             print conversation history\n"
                "  [cyan]/clear[/cyan]               reset conversation history\n"
                "  [cyan]/compact[/cyan]             force context compaction\n"
                "\n"
                "[bold]Sessions[/bold]\n"
                "  [cyan]/save [id][/cyan]           save current session (auto id if omitted)\n"
                "  [cyan]/load <id>[/cyan]           load a named session\n"
                "  [cyan]/tree[/cyan]                list saved sessions\n"
                "  [cyan]/fork <idx>[/cyan]          branch new session from msg idx (see /history)\n"
                "\n"
                "[bold]Shortcuts[/bold]\n"
                "  [magenta]!cmd[/magenta]                run shell command, show output\n"
                "  [magenta]!!cmd[/magenta]               run shell command silently\n"
                "  [magenta]@path/to/file[/magenta]       inline-expand a file into the prompt\n"
                "\n"
                "[bold]Inspection[/bold]\n"
                "  [cyan]/skills[/cyan]              list loaded skill-knowledge packs\n"
                "  [cyan]/cwd[/cyan]                 show the current tool working directory\n"
                "  [cyan]/verbose[/cyan]             toggle verbose mode (raw thinking tokens)"
            ),
            border_style="dim",
            padding=(0, 1),
        ))
        return True

    if cmd == "verbose":
        state.verbose = not state.verbose
        console.print(Text.from_markup(f"[green]✓[/green] verbose: [cyan]{'on' if state.verbose else 'off'}[/cyan]"))
        return True

    if cmd == "permissions":
        if not rest:
            console.print(Text.from_markup(f"mode: [{'yellow' if state.yolo else 'green'}]{'yolo' if state.yolo else 'confirm'}[/]"))
            return True
        mode = rest.lower().strip()
        if mode in ("yolo", "accept-all", "auto"):
            state.yolo = True
            console.print(Text.from_markup("[yellow]✓[/yellow] permissions: yolo — every tool auto-approved"))
        elif mode in ("confirm", "manual", "ask"):
            state.yolo = False
            console.print(Text.from_markup("[green]✓[/green] permissions: confirm — will prompt for dangerous tools"))
        else:
            console.print(Text.from_markup(f"[red]unknown mode: {mode}[/red]  [dim](try: yolo, confirm)[/dim]"))
        return True

    if cmd == "thinking":
        from .services.model_manager import model_manager, MODELS
        from .services import agents as agent_store
        model_id = model_manager.current_model
        if not model_id:
            console.print(Text.from_markup("[dim]no local model active[/dim]"))
            return True
        cfg = MODELS.get(model_id, {})
        if cfg.get("online"):
            console.print(Text.from_markup("[dim]thinking toggle is only for local models[/dim]"))
            return True
        current = model_manager.thinking_enabled
        new = not current
        try:
            prefs = dict(agent_store.read_preferences(agent_store.get_active()))
        except Exception:
            prefs = {}
        vision_enabled = bool(prefs.get("vision_enabled", True))
        ctx_override = prefs.get("ctx")
        try:
            if ctx_override is not None:
                ctx_override = int(ctx_override)
        except Exception:
            ctx_override = None
        with console.status(f"[cyan]{'enabling' if new else 'disabling'} thinking — reloading llama-server…[/cyan]", spinner="dots"):
            try:
                model_manager.start(model_id, vision_enabled, ctx_override, new)
            except Exception as e:
                console.print(Text.from_markup(f"[red]reload failed: {e}[/red]"))
                return True
        try:
            agent_store.update_preferences(agent_store.get_active(), {"thinking_enabled": new})
        except Exception:
            pass
        console.print(Text.from_markup(f"[green]✓[/green] thinking: [cyan]{'on' if new else 'off'}[/cyan]"))
        return True

    if cmd == "history":
        if state.llm is None or not state.llm._history:
            console.print(Text.from_markup("[dim]empty[/dim]"))
            return True
        for i, m in enumerate(state.llm._history):
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            content = str(content or "")[:200].replace("\n", " ")
            tc = m.get("tool_calls") or []
            tc_desc = ""
            if tc:
                names = [t.get("function", {}).get("name", "?") for t in tc]
                tc_desc = f" [tool_calls: {', '.join(names)}]"
            color = {"user": "cyan", "assistant": "magenta", "tool": "dim green", "thinking": "dim yellow"}.get(role, "white")
            console.print(Text.from_markup(f"[{color}]{i:>3}. {role:<10}[/] {content}{tc_desc}"))
        return True

    if cmd == "save":
        if state.llm is None:
            console.print(Text.from_markup("[dim]no history to save[/dim]"))
            return True
        import uuid
        sid = rest.strip() or f"cli_{uuid.uuid4().hex[:8]}"
        try:
            path = state.llm.save_session(sid)
            console.print(Text.from_markup(f"[green]✓[/green] saved [cyan]{sid}[/cyan] [dim]({path})[/dim]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]save failed: {e}[/red]"))
        return True

    if cmd == "load":
        if not rest:
            console.print(Text.from_markup("[red]usage: /load <session_id>[/red]"))
            return True
        state._ensure_llm()
        try:
            ok = state.llm.load_session(rest.strip())
        except Exception as e:
            console.print(Text.from_markup(f"[red]load failed: {e}[/red]"))
            return True
        if ok:
            n = len(state.llm._history)
            console.print(Text.from_markup(f"[green]✓[/green] loaded [cyan]{rest.strip()}[/cyan] ({n} messages)"))
        else:
            console.print(Text.from_markup(f"[red]session not found: {rest.strip()}[/red]"))
        return True

    if cmd == "skills":
        from .skill_knowledge import load_packs
        packs = load_packs()
        if not packs:
            console.print(Text.from_markup("[dim]no skill-knowledge packs loaded[/dim]"))
            return True
        lines = []
        for p in packs:
            triggers = ", ".join(p.triggers[:4]) if p.triggers else "—"
            tools = ", ".join(sorted(p.tools)) if p.tools else "—"
            lines.append(f"[cyan]{p.name:<18}[/cyan]  triggers: [dim]{triggers}[/dim]  tools: [dim]{tools}[/dim]")
        console.print(Panel(Text.from_markup("\n".join(lines)), title="skill packs", border_style="dim", padding=(0, 1)))
        return True

    if cmd == "cwd":
        if state.llm is None:
            console.print(Text.from_markup("[dim]llm not initialized[/dim]"))
            return True
        console.print(Text.from_markup(f"[cyan]{state.llm._cwd}[/cyan]"))
        return True

    if cmd == "tree":
        # List saved sessions for the active agent with message counts + mtime.
        from .services.llm import LLMService
        sessions = LLMService.list_sessions()
        if not sessions:
            console.print(Text.from_markup("[dim]no saved sessions yet[/dim]"))
            return True
        lines = []
        for s in sessions[:40]:
            sid = s.get("id", "")
            title = s.get("title", "Untitled")
            ts = s.get("saved_at", "")[:16].replace("T", " ")
            n = s.get("message_count", 0)
            lines.append(f"[cyan]{sid:<14}[/cyan]  [dim]{ts}[/dim]  [magenta]{n:>3} msgs[/magenta]  {title[:60]}")
        console.print(Panel(Text.from_markup("\n".join(lines)), title="sessions", border_style="dim", padding=(0, 1)))
        return True

    if cmd == "fork":
        # /fork <idx>  — create a new session keeping only messages [0..idx)
        if state.llm is None or not state.llm._history:
            console.print(Text.from_markup("[dim]nothing to fork[/dim]"))
            return True
        try:
            idx = int(rest)
        except (TypeError, ValueError):
            console.print(Text.from_markup("[red]usage: /fork <message_index>[/red]  [dim](see /history for indices)[/dim]"))
            return True
        if not (0 <= idx <= len(state.llm._history)):
            console.print(Text.from_markup(f"[red]index out of range (0..{len(state.llm._history)})[/red]"))
            return True
        state.llm._history = state.llm._history[:idx]
        import uuid
        new_sid = f"fork_{uuid.uuid4().hex[:8]}"
        try:
            path = state.llm.save_session(new_sid)
            console.print(Text.from_markup(f"[green]✓[/green] forked at idx {idx} → [cyan]{new_sid}[/cyan] [dim]({path})[/dim]"))
        except Exception as e:
            console.print(Text.from_markup(f"[red]fork save failed: {e}[/red]"))
        return True

    if cmd == "model":
        if not rest:
            console.print(Text.from_markup(f"[cyan]{state.model_label()}[/cyan]  [dim]({state.backend_label()})[/dim]"))
            return True
        # Switch
        from .services.model_manager import MODELS, resolve_model_id
        try:
            mid = resolve_model_id(rest)
        except ValueError as e:
            console.print(Text.from_markup(f"[red]{e}[/red]"))
            return True
        state.model_alias = rest
        cfg = MODELS[mid]
        if cfg.get("online"):
            from .services import agents as agent_store
            api_key = os.getenv(cfg["api_key_env"], "")
            if not api_key:
                console.print(Text.from_markup(f"[red]Missing {cfg['api_key_env']}[/red]"))
                return True
            os.environ["LLM_BASE_URL"] = cfg["base_url"]
            os.environ["LLM_API_KEY"] = api_key
            os.environ["LLM_MODEL"] = cfg["model_id"]
        else:
            _ensure_llama_started(rest)
        # Re-point the existing LLMService at the new endpoint
        if state.llm is not None:
            state.llm.switch_model(
                os.environ.get("LLM_BASE_URL", ""),
                os.environ.get("LLM_API_KEY", "not-needed"),
                os.environ.get("LLM_MODEL", mid),
            )
        console.print(Text.from_markup(f"[green]✓[/green] switched to [cyan]{state.model_label()}[/cyan]"))
        return True

    if cmd == "agent":
        from .services import agents as agent_store
        sub_parts = rest.split(maxsplit=1) if rest else []
        sub = sub_parts[0].lower() if sub_parts else ""
        sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

        # /agent (no args) — list all agents
        if not rest or sub in ("ls", "list"):
            all_agents = agent_store.list_agents()
            if not all_agents:
                console.print(Text.from_markup("[dim]no agents found[/dim]"))
                return True
            lines = []
            for a in all_agents:
                name = a["name"]
                active = a.get("active", False)
                mark = "[green]→[/green]" if active else " "
                try:
                    prefs = agent_store.read_preferences(name)
                    model = prefs.get("model") or "—"
                    lang = prefs.get("lang") or "en-US"
                except Exception:
                    model, lang = "—", "—"
                try:
                    mcp_count = len(agent_store.read_mcp_servers(name))
                except Exception:
                    mcp_count = 0
                try:
                    tasks_count = len(agent_store.read_scheduled_tasks(name))
                except Exception:
                    tasks_count = 0
                style_name = "cyan bold" if active else "cyan"
                lines.append(
                    f" {mark} [{style_name}]{name:<22}[/{style_name}]  "
                    f"[dim]model:[/dim] {model:<20} "
                    f"[dim]lang:[/dim] {lang:<7} "
                    f"[dim]mcp:[/dim] {mcp_count} "
                    f"[dim]tasks:[/dim] {tasks_count}"
                )
            body = Text.from_markup("\n".join(lines) + "\n\n[dim]/agent <name>        switch\n/agent new <name>    create\n/agent delete <name> delete\n/agent edit [name]   open CLAUDE.md in $EDITOR\n/agent info [name]   full detail[/dim]")
            console.print(Panel(body, title=f"agents ({len(all_agents)})", border_style="dim", padding=(0, 1)))
            return True

        # /agent new <name>
        if sub in ("new", "add", "create"):
            if not sub_arg:
                console.print(Text.from_markup("[red]usage: /agent new <name>[/red]"))
                return True
            try:
                agent_store.create(sub_arg)
            except ValueError as e:
                console.print(Text.from_markup(f"[red]{e}[/red]"))
                return True
            except Exception as e:
                console.print(Text.from_markup(f"[red]create failed: {e}[/red]"))
                return True
            console.print(Text.from_markup(f"[green]✓[/green] created [cyan]{sub_arg}[/cyan] [dim](not yet active — use /agent {sub_arg} to switch)[/dim]"))
            return True

        # /agent delete <name>
        if sub in ("delete", "del", "rm"):
            if not sub_arg:
                console.print(Text.from_markup("[red]usage: /agent delete <name>[/red]"))
                return True
            if sub_arg == "default":
                console.print(Text.from_markup("[red]cannot delete the default agent[/red]"))
                return True
            if not agent_store.exists(sub_arg):
                console.print(Text.from_markup(f"[red]unknown agent: {sub_arg}[/red]"))
                return True
            # Confirm (always, even under --yolo — this is destructive and non-agent-facing)
            console.print(Text.from_markup(
                f"[yellow]? delete agent '[cyan]{sub_arg}[/cyan]' and ALL its sessions, skills, scheduled tasks? [y/N][/yellow] "
            ), end="")
            try:
                answer = sys.stdin.readline().strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                console.print(Text.from_markup("[dim]cancelled[/dim]"))
                return True
            try:
                agent_store.delete(sub_arg)
            except Exception as e:
                console.print(Text.from_markup(f"[red]delete failed: {e}[/red]"))
                return True
            # If it was the active agent, fall back to default
            if state.agent_name == sub_arg:
                agent_store.set_active("default")
                state.agent_name = "default"
                if state.llm is not None:
                    state.llm.set_agent("default")
                console.print(Text.from_markup("[dim]active agent reset to [cyan]default[/cyan][/dim]"))
            console.print(Text.from_markup(f"[green]✓[/green] deleted [cyan]{sub_arg}[/cyan]"))
            return True

        # /agent edit [name] — open CLAUDE.md in $EDITOR
        if sub in ("edit",):
            target = sub_arg or state.agent_name
            if not agent_store.exists(target):
                console.print(Text.from_markup(f"[red]unknown agent: {target}[/red]"))
                return True
            path = agent_store.claude_md_path(target)
            editor = os.getenv("EDITOR") or os.getenv("VISUAL") or ("notepad" if os.name == "nt" else "vi")
            console.print(Text.from_markup(f"[dim]opening [cyan]{path}[/cyan] in [cyan]{editor}[/cyan]…[/dim]"))
            try:
                import subprocess as _sp
                _sp.call([editor, str(path)])
                console.print(Text.from_markup(f"[green]✓[/green] CLAUDE.md for [cyan]{target}[/cyan] saved — will be re-read on next turn"))
            except Exception as e:
                console.print(Text.from_markup(f"[red]editor failed: {e}[/red]  [dim]path: {path}[/dim]"))
            return True

        # /agent info [name]
        if sub in ("info", "show"):
            target = sub_arg or state.agent_name
            if not agent_store.exists(target):
                console.print(Text.from_markup(f"[red]unknown agent: {target}[/red]"))
                return True
            prefs = agent_store.read_preferences(target)
            mcp = agent_store.read_mcp_servers(target)
            tasks = agent_store.read_scheduled_tasks(target)
            claude_md = agent_store.read_claude_md(target)
            lines = [
                f"[bold cyan]{target}[/bold cyan]  {'(active)' if target == state.agent_name else ''}",
                f"[dim]dir:[/dim]       {agent_store.agent_dir(target)}",
                f"[dim]model:[/dim]     {prefs.get('model') or '—'}",
                f"[dim]lang:[/dim]      {prefs.get('lang') or 'en-US'}",
                f"[dim]voice:[/dim]     {prefs.get('voice') or '—'}",
                f"[dim]vision:[/dim]    {prefs.get('vision_enabled', True)}",
                f"[dim]thinking:[/dim]  {prefs.get('thinking_enabled', False)}",
                f"[dim]ctx:[/dim]       {prefs.get('ctx') or '—'}",
                f"[dim]mcp:[/dim]       {len(mcp)} servers" + (f" ({', '.join(mcp)})" if mcp else ""),
                f"[dim]tasks:[/dim]     {len(tasks)} scheduled",
                f"[dim]claude_md:[/dim] {len(claude_md)} chars",
            ]
            console.print(Panel(Text.from_markup("\n".join(lines)), border_style="dim", padding=(0, 1)))
            return True

        # /agent <name> — switch (the default interpretation, kept last)
        target = rest.strip()
        if not agent_store.exists(target):
            console.print(Text.from_markup(f"[red]unknown agent: {target}[/red]  [dim](try /agent with no args to list)[/dim]"))
            return True
        agent_store.set_active(target)
        state.agent_name = target
        if state.llm is not None:
            state.llm.set_agent(target)
        console.print(Text.from_markup(f"[green]✓[/green] active agent: [cyan]{target}[/cyan]"))
        return True

    if cmd == "clear":
        if state.llm is not None:
            state.llm._history.clear()
        console.print(Text.from_markup("[green]✓[/green] history cleared"))
        return True

    if cmd == "compact":
        if state.llm is None:
            console.print(Text.from_markup("[dim]nothing to compact yet[/dim]"))
            return True
        before = len(state.llm._history)
        try:
            await state.llm.auto_compact()
        except Exception as e:
            console.print(Text.from_markup(f"[red]compact failed: {e}[/red]"))
            return True
        after = len(state.llm._history)
        console.print(Text.from_markup(f"[green]✓[/green] compacted [dim]({before} → {after} messages)[/dim]"))
        return True

    if cmd == "context":
        used, mx = state.context_stats()
        if not mx:
            console.print(Text.from_markup("[dim]context window: unknown[/dim]"))
            return True
        pct = int(used / mx * 100) if mx else 0
        n_msgs = len(state.llm._history) if state.llm else 0
        console.print(Panel(
            Text.from_markup(
                f"used: [cyan]{_fmt_ctx(used)}[/cyan] / [cyan]{_fmt_ctx(mx)}[/cyan] ([{'green' if pct<70 else 'yellow' if pct<85 else 'red'}]{pct}%[/])\n"
                f"messages in history: [cyan]{n_msgs}[/cyan]\n"
                f"model: [cyan]{state.model_label()}[/cyan]  [dim]({state.backend_label()})[/dim]"
            ),
            title="context",
            border_style="dim",
            padding=(0, 1),
        ))
        return True

    console.print(Text.from_markup(f"[red]unknown command: /{cmd}[/red]  [dim](try /help)[/dim]"))
    return True


# ──────────────────────────────────────────────────────────────────────────
# llama-server lifecycle
# ──────────────────────────────────────────────────────────────────────────

def _attach_to_running_llama(model_id: str, config: dict, port: int) -> bool:
    """Best-effort probe: if llama-server is already up (e.g. started by the
    web UI running in parallel) and is serving OUR model, reuse it instead of
    kicking off a full restart. Returns True on successful attach."""
    try:
        import httpx
        # Use 127.0.0.1 explicitly — resolving "localhost" first tries IPv6
        # ::1 on Windows, which takes ~2s to time out before falling back to
        # IPv4. The literal keeps the probe under 200ms.
        h = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
        if h.status_code != 200:
            return False
        info = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
        if info.status_code != 200:
            return False
        data = info.json().get("data") or [{}]
        running_id = data[0].get("id", "") if data else ""
        expected = os.path.basename(config.get("model", "") or "")
        if not expected:
            return False
        # The /v1/models id typically contains the GGUF basename. Substring
        # match is enough to detect "same model, same quant".
        if expected not in running_id:
            return False
        os.environ["LLM_MODEL"] = model_id
        os.environ["LLM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        os.environ["LLM_API_KEY"] = "not-needed"
        # Mirror the running state into the singleton so model_manager.current_model,
        # context_size, etc. return the right values without actually re-starting.
        from .services.model_manager import model_manager, MODELS
        model_manager._current_model = model_id
        cfg = MODELS.get(model_id, {})
        if "ctx" in cfg:
            try:
                model_manager._last_ctx = int(cfg["ctx"])
            except (TypeError, ValueError):
                pass
        model_manager._last_vision_enabled = True
        model_manager._last_thinking_enabled = False
        return True
    except Exception:
        return False


def _ensure_llama_started(model_alias: str) -> None:
    """Start llama-server, or attach to an already-running instance.

    Falls back to `9b` if the requested alias is unknown (e.g. a stale
    agent-preference pointing at a model that's since been removed).
    """
    from .services.model_manager import model_manager, MODELS, resolve_model_id
    from .services import agents as agent_store

    try:
        model_id = resolve_model_id(model_alias)
    except ValueError as e:
        console.print(Text.from_markup(f"[yellow]warn:[/yellow] {e}"))
        console.print(Text.from_markup("[yellow]warn:[/yellow] falling back to [cyan]9b[/cyan]"))
        model_alias = "9b"
        model_id = resolve_model_id(model_alias)
    config = MODELS[model_id]

    if config.get("online"):
        api_key = os.getenv(config["api_key_env"], "")
        if not api_key:
            console.print(Text.from_markup(f"[red]Missing {config['api_key_env']} in environment.[/red]"))
            sys.exit(1)
        os.environ["LLM_BASE_URL"] = config["base_url"]
        os.environ["LLM_API_KEY"] = api_key
        os.environ["LLM_MODEL"] = config["model_id"]
        return

    # Attach to any llama-server already running this model (e.g. web UI).
    port = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
    if _attach_to_running_llama(model_id, config, port):
        console.print(Text.from_markup(
            f"[green]✓[/green] attached to running llama-server [dim](model: {os.path.basename(config.get('model', ''))} · port {port})[/dim]"
        ))
        return

    try:
        agent_store.ensure_default()
        prefs = agent_store.read_preferences(agent_store.get_active())
    except Exception:
        prefs = {}
    vision_enabled = bool(prefs.get("vision_enabled", True))
    thinking_enabled = bool(prefs.get("thinking_enabled", False))
    ctx_override = prefs.get("ctx")
    try:
        if ctx_override is not None:
            ctx_override = int(ctx_override)
    except Exception:
        ctx_override = None

    with console.status("[cyan]starting llama-server…[/cyan]", spinner="dots"):
        model_manager.start(model_id, vision_enabled, ctx_override, thinking_enabled)
    os.environ["LLM_MODEL"] = model_id
    os.environ["LLM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    os.environ["LLM_API_KEY"] = "not-needed"


# ──────────────────────────────────────────────────────────────────────────
# REPL + one-shot
# ──────────────────────────────────────────────────────────────────────────

async def _run_one_turn(state: CLIState, prompt: str) -> None:
    global STREAMED_TOKENS_IN_TURN
    state._ensure_llm()
    STREAMED_TOKENS_IN_TURN = 0

    done = asyncio.Event()

    async def _on_done() -> None:
        done.set()

    state.llm._on_done = _on_done  # type: ignore

    # Install a SIGINT handler scoped to this turn: first Ctrl+C asks the LLM
    # to stop gracefully (flips `_running` so the stream loop exits on the next
    # token). Second Ctrl+C raises KeyboardInterrupt to abort hard. Restored to
    # the previous handler on return so the REPL's own Ctrl+C behavior
    # (clear-line in prompt_toolkit) is unaffected.
    import signal
    _interrupts = {"count": 0}

    def _sigint_handler(signum, frame):
        _interrupts["count"] += 1
        if _interrupts["count"] == 1:
            try:
                state.llm._running = False  # type: ignore[attr-defined]
            except Exception:
                pass
            sys.stderr.write("\n\033[33m⏹ interrupt requested — waiting for turn to unwind…\033[0m\n")
            sys.stderr.flush()
        else:
            # Second Ctrl+C — do the default (raise KeyboardInterrupt)
            raise KeyboardInterrupt()

    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    # No spinner during streaming: Rich's Live rendering on stderr competes with
    # raw-bytes token streaming on stdout, which produced visible line breaks
    # between the first two tokens (e.g. "Good" → "\n" → " evening"). The
    # tokens themselves are enough live feedback.
    try:
        await state.llm.start(prompt)
        await done.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Hard-abort path: force-stop and swallow so the REPL continues.
        try:
            state.llm._running = False  # type: ignore[attr-defined]
        except Exception:
            pass
        console.print(Text.from_markup("\n[yellow]⏹ turn aborted[/yellow]"))
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if STREAMED_TOKENS_IN_TURN > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _read_prompt_line() -> Optional[str]:
    """Read one line from the user with readline (history, line editing).

    Returns None on EOF. Empty string on Ctrl-C / interrupt so the caller can
    handle it without exiting the REPL.
    """
    try:
        return input("› ")
    except EOFError:
        return None
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        return ""


async def _run_bash_shortcut(state: CLIState, command: str, silent: bool) -> None:
    """! prefix: run bash and show output. !! prefix: silent."""
    cwd = state.llm._cwd if state.llm else os.getcwd()
    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            console.print(Text.from_markup("[red]! timed out after 60s[/red]"))
            return
    except (asyncio.CancelledError, KeyboardInterrupt):
        console.print(Text.from_markup("[yellow]! interrupted[/yellow]"))
        return
    except Exception as e:
        console.print(Text.from_markup(f"[red]! failed: {e}[/red]"))
        return
    finally:
        # Guarantee the subprocess transport is closed so Windows Proactor
        # doesn't spew ResourceWarning tracebacks at interpreter shutdown.
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
    output = stdout.decode("utf-8", errors="replace")
    rc = proc.returncode
    if silent:
        console.print(Text.from_markup(f"[dim]! (silent) exit={rc} {len(output)} chars[/dim]"))
        return
    if output.strip():
        sys.stdout.write(output)
        sys.stdout.flush()
        if not output.endswith("\n"):
            sys.stdout.write("\n")
    console.print(Text.from_markup(f"[dim]exit={rc}[/dim]"))


_FILE_REF_RE = __import__("re").compile(r"(?:^|(?<=\s))@([A-Za-z0-9_./\\:-]+)")


def _expand_file_refs(state: CLIState, prompt: str) -> str:
    """Replace `@path/to/file` with the file contents wrapped in code fences."""
    def _sub(match):
        rel = match.group(1)
        # Resolve against the agent's cwd (same sandbox as tools)
        base = Path(state.llm._cwd) if state.llm else Path.cwd()
        candidate = (base / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
        try:
            base_resolved = base.resolve()
            if not str(candidate).startswith(str(base_resolved)):
                return match.group(0)  # leave untouched if it escapes the sandbox
        except Exception:
            return match.group(0)
        if not candidate.is_file():
            return match.group(0)
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return match.group(0)
        if len(content) > 32_000:
            content = content[:32_000] + "\n... (truncated)"
        lang = candidate.suffix.lstrip(".")
        return f"`{rel}`:\n```{lang}\n{content}\n```"
    expanded = _FILE_REF_RE.sub(_sub, prompt)
    if expanded != prompt and state.verbose:
        console.print(Text.from_markup("[dim](@file refs expanded)[/dim]"))
    return expanded


async def _handle_keyshortcut(state: CLIState, action: str) -> None:
    """Dispatched from the TUI key bindings (Shift+Tab, Ctrl+L, …)."""
    if action == "thinking_toggle":
        # Two-press confirmation: the first Shift+Tab arms a 5-second window,
        # the second actually toggles. Avoids an accidental llama-server reload
        # (several seconds) from a mistyped keystroke.
        import time as _time
        now = _time.monotonic()
        if state.thinking_toggle_armed_until > now:
            state.thinking_toggle_armed_until = 0.0
            await _handle_slash(state, "/thinking")
        else:
            state.thinking_toggle_armed_until = now + 5.0
            console.print(Text.from_markup(
                "[yellow]? Shift+Tab again within 5s to confirm[/yellow] "
                "[dim](reasoning toggle reloads llama-server ~5s)[/dim]"
            ))
        return
    if action == "pick_model":
        from .services.model_manager import MODELS, MODEL_ALIASES
        lines = []
        for alias, mid in sorted(MODEL_ALIASES.items()):
            cfg = MODELS.get(mid, {})
            mark = "→" if alias == state.model_alias or mid == state.model_alias else " "
            lines.append(f" {mark} [cyan]{alias:<10}[/cyan]  {cfg.get('name', mid)}")
        console.print(Panel(Text.from_markup("\n".join(lines)), title="models — type /model <alias>", border_style="dim", padding=(0, 1)))
        return


async def _repl(state: CLIState) -> None:
    # Prime the cache so the toolbar has real values on its first paint.
    state.refresh_context_stats()
    # Initial status line
    used, mx = state.context_stats()
    _status_line(used, mx, state.model_label())

    # Prefer the prompt_toolkit TUI when available
    from . import tui as _tui
    use_tui = _tui.HAS_TUI
    ptk_prompt = None
    if use_tui:
        try:
            ptk_prompt = _tui.TUIPrompt(
                state=state,
                on_shortcut=lambda action: _handle_keyshortcut(state, action),
            )
        except Exception as e:
            console.print(Text.from_markup(f"[yellow]warn:[/yellow] prompt_toolkit init failed ({e}), falling back to readline"))
            use_tui = False

    while True:
        console.print("")
        try:
            if use_tui and ptk_prompt is not None:
                line = await ptk_prompt.prompt_async()
            else:
                line = await asyncio.to_thread(_read_prompt_line)
        except KeyboardInterrupt:
            console.print("\n[dim]^C — type /quit to exit[/dim]")
            continue
        if line is None:  # EOF (Ctrl-D)
            break
        line = line.strip()
        if not line:
            continue

        # ! / !! shortcut — run bash without invoking LLM
        if line.startswith("!!"):
            await _run_bash_shortcut(state, line[2:].strip(), silent=True)
            continue
        if line.startswith("!") and not line.startswith("!/"):
            await _run_bash_shortcut(state, line[1:].strip(), silent=False)
            continue

        try:
            handled = await _handle_slash(state, line)
        except SystemExit:
            raise
        if handled:
            continue

        # Expand @file references into inline fenced code blocks
        line = _expand_file_refs(state, line)
        await _run_one_turn(state, line)
        # Refresh cached context usage now (NOT per-key in the toolbar).
        state.refresh_context_stats()
        used, mx = state.context_stats()
        # With the TUI the bottom toolbar already shows context; only print the
        # explicit status line in fallback mode.
        if not use_tui:
            _status_line(used, mx, state.model_label())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="ainow",
        description="AINow CLI — headless local agent with the same tools and skill packs as the web UI.",
    )
    p.add_argument("prompt", nargs="?", help="One-shot prompt. If omitted, drops into a REPL unless -i is set.")
    p.add_argument("-i", "--interactive", action="store_true", help="Force interactive REPL after (optional) first prompt.")
    p.add_argument("-m", "--model", default=None, help="Model alias (e.g. 9b, 27b-iq2, online). Defaults to the active agent's last model.")
    p.add_argument("-a", "--agent", default=None, help="Active agent name. Defaults to whatever is currently active.")
    p.add_argument("--no-banner", action="store_true", help="Skip the startup banner.")
    p.add_argument("--minimal-banner", action="store_true", help="One-line banner instead of the boxed panel.")
    p.add_argument("--altscreen", action="store_true", help="Use an alternate screen buffer (TUI owns the terminal). Default: off — keeps normal scrollback so you can scroll back through the conversation.")
    p.add_argument(
        "--yolo", "--auto-approve",
        dest="auto_approve",
        action="store_true",
        help="Skip every confirmation prompt: auto-approve all dangerous tool calls (write, edit, bash, MCP). Use with care.",
    )
    args = p.parse_args(argv)

    # Make imports work when invoked from anywhere
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from .services import agents as agent_store
    agent_store.ensure_default()
    if args.agent:
        if not agent_store.exists(args.agent):
            console.print(Text.from_markup(f"[red]agent '{args.agent}' does not exist.[/red]"))
            sys.exit(1)
        agent_store.set_active(args.agent)

    model_alias = args.model
    if model_alias is None:
        try:
            prefs = agent_store.read_preferences(agent_store.get_active())
            model_alias = prefs.get("model") or "9b"
        except Exception:
            model_alias = "9b"

    _ensure_llama_started(model_alias)

    state = CLIState(yolo=args.auto_approve)
    state.model_alias = model_alias
    state.agent_name = agent_store.get_active()
    # LLMService init is deferred to the first turn — it pulls in ~500ms of
    # openai / httpx imports we don't need to pay for just to render the banner.

    # Expose state to module-level callbacks (tool-result rendering, thinking).
    global _CURRENT_STATE
    _CURRENT_STATE = state

    # Banner — skipped here if we're about to swap to alt-screen (the banner
    # is re-rendered as the first thing on the alt screen so the user sees it).
    _will_altscreen = (
        args.interactive and args.altscreen
        and sys.stdout.isatty() and sys.stderr.isatty()
    )
    if not args.no_banner and not _will_altscreen:
        from .services.model_manager import model_manager
        _banner(
            model_label=state.model_label(),
            backend=state.backend_label(),
            permissions="yolo" if args.auto_approve else "confirm",
            ctx=model_manager.get_context_size(),
            agent=state.agent_name,
            minimal=args.minimal_banner,
        )

    async def _drive() -> None:
        if args.prompt:
            await _run_one_turn(state, args.prompt)
            if args.interactive:
                console.print("")
                await _repl(state)
        else:
            await _repl(state)

    # Alt-screen opt-in via --altscreen. Default is off so the user keeps the
    # terminal's scrollback (Shift+PgUp, mouse wheel) to review past turns.
    _in_altscreen = (
        args.interactive and args.altscreen
        and sys.stdout.isatty() and sys.stderr.isatty()
    )
    if _in_altscreen:
        sys.stdout.write("\033[?1049h\033[2J\033[H")
        sys.stdout.flush()

    # Re-draw the banner after the clear so it's the first thing on the alt screen.
    if _in_altscreen and not args.no_banner:
        from .services.model_manager import model_manager
        _banner(
            model_label=state.model_label(),
            backend=state.backend_label(),
            permissions="yolo" if args.auto_approve else "confirm",
            ctx=model_manager.get_context_size(),
            agent=state.agent_name,
            minimal=args.minimal_banner,
        )

    try:
        asyncio.run(_drive())
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)
    except SystemExit:
        raise
    finally:
        if _in_altscreen:
            # Restore the user's scrollback.
            sys.stdout.write("\033[?1049l")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
