"""Textual-based full-screen TUI for AINow.

Uses Textual (https://textual.textualize.io) — a modern Python TUI framework
with differential rendering, CSS-driven layout, and proper cross-platform
terminal handling.

Layout:
  * Header (branded banner)
  * VerticalScroll chat area (mounts one ChatMessage per user/assistant/tool turn)
  * Footer (status line: cwd, agent, model, ctx, session tok/s)
  * Input at the bottom

The key win over the prompt_toolkit full-screen attempt: Textual handles the
differential rendering + alt-screen + Windows terminal quirks reliably, so the
input/footer stay anchored and tokens stream cleanly into a scrollable area.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical, VerticalScroll
    from textual.reactive import reactive
    from textual.screen import ModalScreen
    from textual.suggester import Suggester
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        Select,
        Static,
        Switch,
    )
    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False


HAS_TEXTUAL = _HAS_TEXTUAL


# ─── Slash-command registry ───────────────────────────────────────────────
# (name, args_hint, description). Used by:
#   * SlashSuggester  → ghost-text autocomplete on the Input
#   * HelpModal       → arrow-navigable list
SLASH_COMMANDS: list[tuple[str, str, str]] = [
    ("/help", "", "show commands in an arrow-navigable list"),
    ("/config", "", "open the model/vision/thinking/ctx config screen"),
    ("/model", "[alias]", "switch model or list available ones"),
    ("/agent", "[sub]", "list agents or switch (/agent <name>)"),
    ("/thinking", "", "toggle reasoning mode (reloads llama-server)"),
    ("/permissions", "[yolo|confirm]", "switch permission mode"),
    ("/context", "", "detailed context-usage breakdown"),
    ("/history", "", "print conversation history"),
    ("/clear", "", "wipe history + screen"),
    ("/compact", "", "force a context compaction"),
    ("/save", "[id]", "save current session"),
    ("/load", "<id>", "load a named session"),
    ("/tree", "", "list saved sessions"),
    ("/fork", "<idx>", "branch a new session from a message index"),
    ("/skills", "", "list loaded skill-knowledge packs"),
    ("/cwd", "", "show the current tool working directory"),
    ("/verbose", "", "toggle verbose mode (raw thinking tokens)"),
    ("/quit", "", "leave AINow"),
    ("/exit", "", "leave AINow"),
]


# Cached git-branch lookup — same as in tui_fullscreen
_BRANCH_CACHE: dict = {}
_BRANCH_TTL = 5.0


def _git_branch_cached(cwd: str) -> str:
    now = time.monotonic()
    hit = _BRANCH_CACHE.get(cwd)
    if hit and now - hit[0] < _BRANCH_TTL:
        return hit[1]
    branch = ""
    try:
        p = Path(cwd)
        for candidate in [p] + list(p.parents):
            head = candidate / ".git" / "HEAD"
            if head.exists():
                line = head.read_text(encoding="utf-8", errors="replace").strip()
                if line.startswith("ref: refs/heads/"):
                    branch = line[len("ref: refs/heads/"):]
                else:
                    branch = line[:8]
                break
    except Exception:
        pass
    _BRANCH_CACHE[cwd] = (now, branch)
    return branch


if _HAS_TEXTUAL:

    class ChatMessage(Static):
        """One message bubble. Can be appended to while streaming."""

        DEFAULT_CSS = """
        ChatMessage {
            padding: 0 1;
            margin: 0 0 1 0;
            height: auto;
        }
        ChatMessage.user {
            color: $primary;
            text-style: bold;
        }
        ChatMessage.assistant { color: $text; }
        ChatMessage.tool { color: $success; }
        ChatMessage.error { color: $error; }
        ChatMessage.system { color: $text-muted; text-style: italic; }
        ChatMessage.thinking { color: $warning; text-style: italic; }
        """

        def __init__(self, role: str, text: str = "", markup: bool = False):
            super().__init__(text, markup=markup)
            self.role = role
            self.add_class(role)
            self._buffer = text

        def append(self, text: str) -> None:
            self._buffer += text
            self.update(self._buffer)

        def set_text(self, text: str) -> None:
            self._buffer = text
            self.update(self._buffer)


    class ThinkingSpinner(Static):
        """Animated 'AINow is thinking…' indicator, shown from user submit
        until the first token arrives. Cycles through braille spinner chars
        every ~100ms; removed from the chat tree on first token / tool call."""

        DEFAULT_CSS = """
        ThinkingSpinner {
            padding: 0 1;
            margin: 0 0 1 0;
            height: 1;
            color: $warning;
            text-style: italic;
        }
        """

        FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        PHRASES = (
            "AINow is thinking",
            "consulting the silicon oracle",
            "warming up the KV cache",
            "decoding",
            "asking the model nicely",
            "crunching tokens",
        )

        def __init__(self) -> None:
            super().__init__("", markup=True)
            self._frame = 0
            import random as _r
            self._phrase = _r.choice(self.PHRASES)
            self._timer = None

        def on_mount(self) -> None:
            self._render_frame()
            # 80ms → 12.5 fps, smooth without wasting CPU
            self._timer = self.set_interval(0.08, self._tick)

        def _tick(self) -> None:
            self._frame = (self._frame + 1) % len(self.FRAMES)
            self._render_frame()

        def _render_frame(self) -> None:
            ch = self.FRAMES[self._frame]
            dots = "." * ((self._frame // 3) % 4)
            self.update(f"[yellow]{ch}[/yellow] [italic dim]{self._phrase}{dots}[/italic dim]")


    class AINowInput(Input):
        """Input subclass with three tweaks over the stock Textual Input:

        1. Tab accepts the suggester's ghost text (via cursor_right, which
           swaps the suggestion into the value when the cursor is at end).
        2. Multi-line pastes (bracketed paste / Ctrl+V) are flattened:
           newlines become single spaces and the full content is preserved.
           Textual's default paste handler only keeps the first splitlines()
           line, so clipboards starting with a newline or containing
           intended multi-line content would silently drop most of the text.
        3. Right-click pastes from the system clipboard (Textual captures
           the mouse before Windows Terminal can handle right-click-paste).
        """

        BINDINGS = [
            Binding("tab", "cursor_right", "Accept suggestion", show=False),
        ]

        def _on_paste(self, event) -> None:
            # Intercept BEFORE the parent Input's default handler runs.
            # prevent_default() stops Textual from calling the base class's
            # _on_paste (which would insert splitlines()[0] again, causing
            # duplicated content). stop() prevents bubbling up the tree.
            from textual import events as _events
            if not isinstance(event, _events.Paste):
                return
            event.prevent_default()
            event.stop()
            text = event.text or ""
            if text:
                self._insert_flattened(text)

        def on_click(self, event) -> None:
            """Right-click is context-sensitive:

            * If there was an active selection at MouseDown, the App
              already copied it (see AINowApp.on_mouse_down) and set
              `_just_copied_selection`. We consume the flag and skip
              paste so you don't accidentally overwrite the clipboard.
            * Otherwise → PASTE the clipboard content into the Input."""
            if getattr(event, "button", 0) != 3:
                return
            event.stop()
            # Did the App already copy a selection on MouseDown?
            app = self.app
            if getattr(app, "_just_copied_selection", False):
                app._just_copied_selection = False
                return
            # No selection → paste.
            try:
                import pyperclip
                text = pyperclip.paste()
            except Exception:
                return
            if text:
                self._insert_flattened(text)

        def _insert_flattened(self, text: str) -> None:
            # Collapse CRLF/CR/LF and runs of whitespace so "\n\n" doesn't
            # become a visible double-space and indented content stays tidy.
            import re as _re
            flat = _re.sub(r"\s+", " ", text).strip()
            if flat:
                self.insert_text_at_cursor(flat)


    class SlashSuggester(Suggester):
        """Ghost-text suggester for `/` commands on the Input widget.

        Matches against the SLASH_COMMANDS registry. Only fires when the
        value starts with `/` and there's no trailing space (i.e. user is
        still typing the command name, not its argument).
        """

        def __init__(self) -> None:
            super().__init__(case_sensitive=False, use_cache=False)

        async def get_suggestion(self, value: str) -> Optional[str]:
            if not value.startswith("/"):
                return None
            if " " in value:
                return None
            prefix = value.lower()
            for name, _hint, _desc in SLASH_COMMANDS:
                if name.lower().startswith(prefix) and name.lower() != prefix:
                    return name
            return None


    class HelpModal(ModalScreen):
        """Arrow-navigable list of slash commands. Enter prefills the input,
        Escape dismisses. Clicking a row also selects it."""

        DEFAULT_CSS = """
        HelpModal {
            align: center middle;
        }
        HelpModal > Vertical {
            background: $surface;
            border: heavy $primary-lighten-3;
            width: 80;
            max-width: 90%;
            height: auto;
            max-height: 80%;
            padding: 1 2;
        }
        HelpModal #help-title {
            text-style: bold;
            color: $primary;
            margin: 0 0 1 0;
        }
        HelpModal ListView {
            height: auto;
            max-height: 20;
            background: $surface;
            border: none;
        }
        HelpModal ListItem {
            padding: 0 1;
        }
        HelpModal ListItem.--highlight {
            background: $primary 30%;
        }
        HelpModal #help-hint {
            margin: 1 0 0 0;
            color: $text-muted;
        }
        """

        BINDINGS = [
            Binding("escape", "dismiss_modal", "Close", priority=True),
            Binding("q", "dismiss_modal", "Close", show=False),
        ]

        def compose(self) -> ComposeResult:
            items: list[ListItem] = []
            for name, hint, desc in SLASH_COMMANDS:
                label = Static(
                    f"[cyan bold]{name:<14}[/cyan bold] [dim]{hint:<18}[/dim] {desc}",
                    markup=True,
                )
                li = ListItem(label)
                li.cmd_name = name  # attach so on_list_view_selected can read it
                items.append(li)
            with Vertical():
                yield Static("AINow — slash commands", id="help-title", markup=True)
                yield ListView(*items, id="help-list")
                yield Static(
                    "[dim]↑/↓ navigate · Enter to prefill · Esc to close[/dim]",
                    id="help-hint",
                    markup=True,
                )

        def on_mount(self) -> None:
            try:
                self.query_one("#help-list", ListView).focus()
            except Exception:
                pass

        def action_dismiss_modal(self) -> None:
            self.dismiss(None)

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            name = getattr(event.item, "cmd_name", None)
            self.dismiss(name)


    class ConfirmModal(ModalScreen):
        """Yes/No modal for dangerous tool-call approval in confirm mode.

        CRITICAL: this replaces the blocking `sys.stdin.readline()` path from
        cli.py, which freezes the Textual event loop (Textual owns stdin in
        alt-screen, so the REPL can never receive the keypress and even
        Ctrl+C can't get through). The modal dismisses with True/False which
        the awaiting `CLIState.confirm` coroutine returns."""

        DEFAULT_CSS = """
        ConfirmModal {
            align: center middle;
        }
        ConfirmModal > Vertical {
            background: $surface;
            border: heavy $warning;
            width: 70;
            max-width: 90%;
            height: auto;
            padding: 1 2;
        }
        ConfirmModal #cf-title {
            text-style: bold;
            color: $warning;
            margin: 0 0 1 0;
        }
        ConfirmModal #cf-detail {
            color: $text-muted;
            margin: 0 0 1 0;
        }
        ConfirmModal #cf-buttons {
            height: 3;
            align-horizontal: right;
            margin: 1 0 0 0;
        }
        ConfirmModal Button { margin: 0 0 0 1; }
        """

        # priority=True so the Screen-level Enter/y/n bindings intercept the
        # key *before* the focused Button's own built-in enter→press binding
        # runs. Otherwise Button swallows Enter and the dismiss(...) from the
        # Pressed handler doesn't reliably fire through the modal.
        BINDINGS = [
            Binding("escape", "deny", "Deny", priority=True),
            Binding("n", "deny", "No", show=False, priority=True),
            Binding("y", "approve", "Yes", show=False, priority=True),
            Binding("enter", "approve", "Approve", show=False, priority=True),
        ]

        def __init__(self, *, name: str, detail: str):
            super().__init__()
            self._name = name
            self._detail = detail

        def compose(self) -> ComposeResult:
            with Vertical():
                yield Static(
                    f"? confirm tool call: [cyan bold]{self._name}[/cyan bold]",
                    id="cf-title",
                    markup=True,
                )
                yield Static(f"[dim]{self._detail}[/dim]", id="cf-detail", markup=True)
                yield Static(
                    "[dim]Enter / y = approve · Esc / n = deny[/dim]",
                    markup=True,
                )
                with Horizontal(id="cf-buttons"):
                    yield Button("Deny", id="cf-deny")
                    yield Button("Approve", id="cf-approve", variant="warning")

        # Intentionally do NOT focus the Approve button on mount — if it has
        # focus, Textual routes Enter to its internal `press` binding, which
        # competes with the Screen-level enter→approve binding and the result
        # is flaky (the user reported "Enter does nothing, y works"). Leaving
        # focus on the screen itself makes the priority bindings reliable.

        def action_approve(self) -> None:
            self.dismiss(True)

        def action_deny(self) -> None:
            self.dismiss(False)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(event.button.id == "cf-approve")


    class ConfigModal(ModalScreen):
        """One-shot model + vision + thinking + ctx config screen.

        Dismisses with either None (cancel) or a dict describing the chosen
        state — AINowApp applies it via model_manager.start() in a worker.
        """

        DEFAULT_CSS = """
        ConfigModal {
            align: center middle;
        }
        ConfigModal > Vertical {
            background: $surface;
            border: heavy $primary-lighten-3;
            width: 78;
            max-width: 90%;
            height: auto;
            padding: 1 2;
        }
        ConfigModal #cfg-title {
            text-style: bold;
            color: $primary;
            margin: 0 0 1 0;
        }
        ConfigModal .row {
            height: 3;
            margin: 0 0 1 0;
        }
        ConfigModal .row Label {
            width: 16;
            padding: 1 1 0 0;
        }
        ConfigModal Select { width: 1fr; }
        ConfigModal #cfg-buttons {
            height: 3;
            align-horizontal: right;
            margin: 1 0 0 0;
        }
        ConfigModal Button { margin: 0 0 0 1; }
        ConfigModal #cfg-hint {
            color: $text-muted;
        }
        """

        # Tab / Shift+Tab as priority=True so the app-level `shift+tab` =>
        # thinking_mode binding doesn't steal focus cycling inside the modal.
        BINDINGS = [
            Binding("escape", "dismiss_modal", "Cancel", priority=True),
            Binding("tab", "focus_next", "Next field", show=False, priority=True),
            Binding("shift+tab", "focus_previous", "Prev field", show=False, priority=True),
        ]

        def __init__(self, *, current: dict):
            super().__init__()
            self._current = current

        def compose(self) -> ComposeResult:
            from .services.model_manager import MODEL_ALIASES, MODELS

            model_options: list[tuple[str, str]] = []
            for alias in sorted(MODEL_ALIASES.keys()):
                mid = MODEL_ALIASES[alias]
                cfg = MODELS.get(mid, {})
                name = cfg.get("name", mid)
                online = " (online)" if cfg.get("online") else ""
                model_options.append((f"{alias:<12}  {name}{online}", alias))

            # Normalize the current value — state.model_alias may hold either
            # the short alias ("9b") or the full model_id ("qwen3.5-9b").
            # Select rejects values that aren't in its options list.
            current_alias = self._current.get("alias")
            allowed_values = {v for _, v in model_options}
            if current_alias and current_alias not in allowed_values:
                reverse = {mid: alias for alias, mid in MODEL_ALIASES.items()}
                current_alias = reverse.get(current_alias)

            ctx_choices = [
                ("4K (4096)", "4096"),
                ("8K (8192)", "8192"),
                ("16K (16384)", "16384"),
                ("32K (32768)", "32768"),
                ("64K (65536)", "65536"),
                ("128K (131072)", "131072"),
                ("256K (262144)", "262144"),
                ("512K (524288)", "524288"),
                ("1M (1048576)", "1048576"),
            ]
            cur_ctx = str(self._current.get("ctx", "") or "")
            if cur_ctx and cur_ctx not in [v for _, v in ctx_choices]:
                ctx_choices.append((f"current ({cur_ctx})", cur_ctx))

            with Vertical():
                yield Static("AINow — session config", id="cfg-title", markup=True)

                with Horizontal(classes="row"):
                    yield Label("model")
                    yield Select(
                        model_options,
                        value=current_alias if current_alias else Select.BLANK,
                        allow_blank=True,
                        id="cfg-model",
                    )

                with Horizontal(classes="row"):
                    yield Label("context")
                    yield Select(
                        ctx_choices,
                        value=cur_ctx if cur_ctx else Select.BLANK,
                        allow_blank=True,
                        id="cfg-ctx",
                    )

                with Horizontal(classes="row"):
                    yield Label("vision")
                    yield Switch(value=bool(self._current.get("vision", True)), id="cfg-vision")

                with Horizontal(classes="row"):
                    yield Label("thinking")
                    yield Switch(value=bool(self._current.get("thinking", False)), id="cfg-thinking")

                yield Static(
                    "[dim]Tab / Shift+Tab to cycle fields · ↑/↓ within a field · "
                    "Apply reloads llama-server · Esc to cancel.[/dim]",
                    id="cfg-hint",
                    markup=True,
                )

                with Horizontal(id="cfg-buttons"):
                    yield Button("Cancel", id="cfg-cancel")
                    yield Button("Apply", id="cfg-apply", variant="primary")

        def action_dismiss_modal(self) -> None:
            self.dismiss(None)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cfg-cancel":
                self.dismiss(None)
                return
            if event.button.id == "cfg-apply":
                model_sel = self.query_one("#cfg-model", Select).value
                ctx_sel = self.query_one("#cfg-ctx", Select).value
                vision = self.query_one("#cfg-vision", Switch).value
                thinking = self.query_one("#cfg-thinking", Switch).value
                try:
                    ctx_val = int(ctx_sel) if ctx_sel and ctx_sel != Select.BLANK else None
                except (TypeError, ValueError):
                    ctx_val = None
                if model_sel == Select.BLANK:
                    model_sel = self._current.get("alias")
                self.dismiss({
                    "alias": model_sel,
                    "ctx": ctx_val,
                    "vision": bool(vision),
                    "thinking": bool(thinking),
                })


    class StatusBar(Static):
        """One-line status bar showing model, ctx%, permissions, session stats."""

        DEFAULT_CSS = """
        StatusBar {
            height: 1;
            background: $boost;
            color: $text;
            padding: 0 1;
        }
        """

        def __init__(self, *, state, **kwargs):
            # Pre-render an initial status string at construction time so the
            # bar is always visible, even if on_mount hasn't fired yet.
            super().__init__("[dim]AINow · loading…[/dim]", markup=True, **kwargs)
            self._state = state

        def on_mount(self):
            self.set_interval(1.0, self.refresh_status)
            # Render something IMMEDIATELY so the bar is visible even if the
            # state isn't fully populated yet.
            self.update("[dim]loading…[/dim]")
            self.refresh_status()

        def refresh_status(self) -> None:
            try:
                s = self._state
                used, mx = s.context_stats()
                pct = int(used / mx * 100) if mx else 0
                pct_color = "green" if pct < 70 else "yellow" if pct < 85 else "red"
                perm = "yolo" if s.yolo else "confirm"
                perm_color = "yellow" if s.yolo else "green"
                cwd = Path(s.llm._cwd).resolve() if s.llm and s.llm._cwd else Path.cwd()
                try:
                    home = Path.home().resolve()
                    try:
                        short = "~/" + str(cwd.relative_to(home)).replace("\\", "/")
                    except ValueError:
                        short = str(cwd)
                except Exception:
                    short = str(cwd)
                branch = _git_branch_cached(str(cwd))
                branch_txt = f" ([white]{branch}[/white])" if branch else ""

                model_name = s.model_label()
                if len(model_name) > 40:
                    model_name = model_name[:39] + "…"
                try:
                    from .services.model_manager import model_manager as _mm
                    reasoning = "high" if _mm.thinking_enabled else "off"
                except Exception:
                    reasoning = "off"
                mx_fmt = f"{mx // 1024}K" if mx and mx >= 1024 else (f"{mx}" if mx else "?")

                sess = ""
                sess_tok = getattr(s, "session_total_tokens", 0)
                sess_sec = getattr(s, "session_total_seconds", 0.0)
                if sess_tok > 0 and sess_sec > 0:
                    avg = sess_tok / sess_sec
                    sess_fmt = f"{sess_tok}" if sess_tok < 1000 else f"{sess_tok // 1000}.{(sess_tok % 1000) // 100}k"
                    sess = f"  ·  [white]{sess_fmt} tok @ {avg:.1f} tok/s[/white]"

                self.update(
                    f"[cyan]{short}[/cyan]{branch_txt}  ·  [dim]agent[/dim] [cyan]{s.agent_name}[/cyan]  "
                    f"·  [dim]ctx[/dim] [{pct_color}]{pct}%/{mx_fmt}[/{pct_color}]  "
                    f"·  [{perm_color}]{perm}[/{perm_color}]{sess}  "
                    f"·  [cyan bold]{model_name}[/cyan bold]  ·  [dim]reason[/dim] [white]{reasoning}[/white]"
                )
            except Exception as e:
                self.update(f"[red]statusbar error: {e}[/red]")


    class AINowApp(App):
        """Textual app for the AINow CLI."""

        CSS = """
        Screen {
            background: $background;
            layout: vertical;
        }

        #chat {
            height: 1fr;
            padding: 1 2 0 2;
            overflow-y: auto;
            background: $surface;
        }

        #input {
            height: 3;
            margin: 0;
            border: heavy $primary-lighten-3;
            padding: 0 1;
        }

        Input {
            background: $background;
        }
        """

        BINDINGS = [
            Binding("ctrl+c", "interrupt_or_quit", "Interrupt / Quit", priority=True),
            Binding("ctrl+d", "quit", "Exit", priority=True),
            Binding("ctrl+o", "toggle_tool_output", "Tool out"),
            Binding("ctrl+t", "toggle_thinking", "Think"),
            Binding("shift+tab", "thinking_mode", "Reason"),
            Binding("ctrl+l", "pick_model", "Model"),
            # NOT priority — otherwise they eat up/down inside pushed modals
            # (e.g. the /help ListView could never navigate). Input has no
            # up/down binding of its own, so these still fire when the Input
            # is focused and bubble up from the Screen.
            Binding("up", "history_prev", "History ↑", show=False),
            Binding("down", "history_next", "History ↓", show=False),
            Binding("pageup", "scroll_up", "Scroll up", show=False),
            Binding("pagedown", "scroll_down", "Scroll down", show=False),
            # F7 because Ctrl+Shift+C is intercepted by Windows Terminal and
            # Ctrl+C is already bound to interrupt/quit. F-keys aren't
            # eaten by any terminal emulator.
            Binding("f7", "copy_chat", "Copy chat"),
        ]

        # Persisted history file — same path as the prompt_toolkit REPL mode
        # so both share a single command history.
        HISTORY_FILE = str(Path(os.path.expanduser("~")) / ".ainow_cli_history")

        def __init__(
            self,
            *,
            state,
            submit_handler: Callable[[str], Awaitable[None]],
            slash_handler: Callable[[str], Awaitable[bool]],
            bash_shortcut_handler: Callable[[str, bool], Awaitable[None]],
            keyshortcut_handler: Callable[[str], Awaitable[None]],
        ):
            super().__init__()
            self.state = state
            self._submit = submit_handler
            self._slash = slash_handler
            self._bash = bash_shortcut_handler
            self._keyshortcut = keyshortcut_handler
            self._turn_task: Optional[asyncio.Task] = None
            self._current_assistant_msg: Optional[ChatMessage] = None
            self._current_thinking_msg: Optional[ChatMessage] = None
            self._spinner: Optional[ThinkingSpinner] = None
            # Right-click copy flag: set on MouseDown (before Textual's
            # default handler clears the selection on MouseUp), consumed
            # on the subsequent Click so the Input knows to skip paste.
            self._just_copied_selection: bool = False
            # Input history — loaded from disk on startup, appended on submit,
            # persisted on exit.
            self._history: list[str] = []
            self._history_idx: Optional[int] = None  # None = editing fresh input
            self._history_saved_text: str = ""  # buffer the unsubmitted input
            self._load_history()

        # --- layout -------------------------------------------------------

        def compose(self) -> ComposeResult:
            yield VerticalScroll(id="chat")
            yield StatusBar(state=self.state)
            yield AINowInput(
                placeholder="Type a message, /help for commands, Ctrl+D to exit…",
                id="input",
                suggester=SlashSuggester(),
            )

        async def on_mount(self) -> None:
            # Banner as the first message.
            s = self.state
            try:
                from .services.model_manager import model_manager as _mm
                ctx = _mm.get_context_size()
                vision_on = bool(_mm.vision_enabled)
                thinking_on = bool(_mm.thinking_enabled)
            except Exception:
                ctx = 0
                vision_on, thinking_on = True, False
            ctx_fmt = f"{ctx // 1024}K" if ctx >= 1024 else (f"{ctx}" if ctx else "unknown")
            vision_color = "green" if vision_on else "dim"
            thinking_color = "green" if thinking_on else "dim"
            banner = (
                f"[bold magenta]AINow[/bold magenta]  [dim]v1.1[/dim]\n"
                f"[italic]Local-first AI agent framework — chat, code, voice, vision[/italic]\n\n"
                f"[dim]model:[/dim]       [cyan]{s.model_label()}[/cyan]  [dim]({s.backend_label()})[/dim]\n"
                f"[dim]agent:[/dim]       [cyan]{s.agent_name}[/cyan]\n"
                f"[dim]permissions:[/dim] [{'yellow' if s.yolo else 'green'}]{'yolo' if s.yolo else 'confirm'}[/]\n"
                f"[dim]context:[/dim]     [cyan]{ctx_fmt}[/cyan]\n"
                f"[dim]vision:[/dim]      [{vision_color}]{'on' if vision_on else 'off'}[/{vision_color}]\n"
                f"[dim]thinking:[/dim]    [{thinking_color}]{'on' if thinking_on else 'off'}[/{thinking_color}]\n\n"
                f"[dim]/help for commands  ·  /config to tune  ·  Ctrl+D to exit[/dim]\n"
                f"[dim][white]F7[/white] copies the chat transcript to the clipboard[/dim]"
            )
            self._append_message("system", banner, markup=True)
            self.query_one("#input", Input).focus()

        # --- helpers ------------------------------------------------------

        def _append_message(self, role: str, text: str, markup: bool = False) -> ChatMessage:
            msg = ChatMessage(role, text, markup=markup)
            chat = self.query_one("#chat", VerticalScroll)
            chat.mount(msg)
            chat.scroll_end(animate=False)
            return msg

        def _scroll_to_end(self) -> None:
            try:
                self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
            except Exception:
                pass

        # --- ChatLog-style API the CLI callbacks can use -----------------

        def log_user(self, text: str) -> None:
            self._append_message("user", f"› {text}")

        def log_tool_arrow(self, name: str, arg_summary: str) -> None:
            self._dismiss_spinner()
            self._append_message(
                "tool", f"[cyan]→[/cyan] [bold]{name}[/bold]  [dim]{arg_summary}[/dim]",
                markup=True,
            )

        def log_tool_result(self, name: str, mark: str, detail: str, is_error: bool) -> None:
            color = "red" if is_error else "green"
            self._append_message(
                "tool",
                f"[{color}]{mark}[/{color}] [cyan]→[/cyan] [bold]{name}[/bold]  [dim]{detail}[/dim]",
                markup=True,
            )

        def log_system(self, text: str, style: str = "dim") -> None:
            # Empty style would produce "[][/]" — Rich parses "[/]" as an
            # auto-closing tag with nothing to close and crashes. Skip the
            # wrapping tag entirely when style is falsy.
            if style:
                text = f"[{style}]{text}[/{style}]"
            self._append_message("system", text, markup=True)

        def log_error(self, text: str) -> None:
            self._append_message("error", f"[red]{text}[/red]", markup=True)

        def log_markup(self, rich_markup: str) -> None:
            self._append_message("system", rich_markup, markup=True)

        def token_append(self, token: str) -> None:
            """Append a streaming token to the current assistant message."""
            self._dismiss_spinner()
            if self._current_assistant_msg is None:
                self._current_assistant_msg = ChatMessage("assistant", "", markup=False)
                self.query_one("#chat", VerticalScroll).mount(self._current_assistant_msg)
            self._current_assistant_msg.append(token)
            self._scroll_to_end()

        def finalize_turn(self) -> None:
            """Mark the current assistant message as complete. Next token starts a new bubble."""
            self._dismiss_spinner()
            self._current_assistant_msg = None
            self._current_thinking_msg = None

        def thinking_append(self, text: str) -> None:
            """Stream a chunk of reasoning_content into a dedicated
            thinking bubble. Only called when the user opted in via
            `/verbose` or Ctrl+T (see _on_thinking)."""
            self._dismiss_spinner()
            if self._current_thinking_msg is None:
                self._current_thinking_msg = ChatMessage(
                    "thinking", "[dim italic]💭 thinking…[/dim italic]\n", markup=True,
                )
                self.query_one("#chat", VerticalScroll).mount(self._current_thinking_msg)
            self._current_thinking_msg.append(text)
            self._scroll_to_end()

        def thinking_finalize(self, duration: float) -> None:
            """Close out the thinking bubble with a duration footer."""
            if self._current_thinking_msg is not None:
                self._current_thinking_msg.append(
                    f"\n[dim italic]— thought for {duration:.1f}s —[/dim italic]"
                )
                self._current_thinking_msg = None
            self._scroll_to_end()

        def show_spinner(self) -> None:
            """Mount the thinking animation right after the user message."""
            if self._spinner is not None:
                return
            self._spinner = ThinkingSpinner()
            self.query_one("#chat", VerticalScroll).mount(self._spinner)
            self._scroll_to_end()

        def _dismiss_spinner(self) -> None:
            if self._spinner is not None:
                try:
                    self._spinner.remove()
                except Exception:
                    pass
                self._spinner = None

        # --- history -----------------------------------------------------

        def _load_history(self) -> None:
            try:
                with open(self.HISTORY_FILE, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if line:
                            self._history.append(line)
            except FileNotFoundError:
                pass
            except Exception:
                pass

        def _append_history(self, line: str) -> None:
            if not line.strip():
                return
            # De-dupe consecutive identical entries
            if self._history and self._history[-1] == line:
                return
            self._history.append(line)
            try:
                with open(self.HISTORY_FILE, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

        def _apply_history_entry(self) -> None:
            inp = self.query_one("#input", Input)
            if self._history_idx is None:
                inp.value = self._history_saved_text
            else:
                inp.value = self._history[self._history_idx]
            try:
                inp.cursor_position = len(inp.value)
            except Exception:
                pass

        def action_history_prev(self) -> None:
            if not self._history:
                return
            try:
                inp = self.query_one("#input", Input)
            except Exception:
                return
            if not inp.has_focus:
                return
            if self._history_idx is None:
                self._history_saved_text = inp.value
                self._history_idx = len(self._history) - 1
            elif self._history_idx > 0:
                self._history_idx -= 1
            self._apply_history_entry()

        def action_history_next(self) -> None:
            if self._history_idx is None:
                return
            try:
                inp = self.query_one("#input", Input)
            except Exception:
                return
            if not inp.has_focus:
                return
            if self._history_idx < len(self._history) - 1:
                self._history_idx += 1
            else:
                self._history_idx = None  # back to the unsubmitted buffer
            self._apply_history_entry()

        # --- input submission --------------------------------------------

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            event.input.value = ""
            # Reset history cursor + clear the saved buffer on every submit
            self._history_idx = None
            self._history_saved_text = ""
            if not text:
                return
            self._append_history(text)
            if self._turn_task and not self._turn_task.done():
                self.log_system("a turn is already running — press Ctrl+C to interrupt", "yellow")
                return
            self.log_user(text)
            # Spinner is owned by _handle_line — only shown for actual LLM
            # calls, NOT slash commands or bash shortcuts (those ran before
            # and left the spinner hanging, making it look like the LLM was
            # still "thinking" after the command already completed).
            self._turn_task = asyncio.create_task(self._handle_line(text))

        async def _handle_line(self, line: str) -> None:
            if line.startswith("!!"):
                await self._bash(line[2:].strip(), True)
                return
            if line.startswith("!") and not line.startswith("!/"):
                await self._bash(line[1:].strip(), False)
                return
            try:
                handled = await self._slash(line)
            except SystemExit:
                self.exit()
                return
            if handled:
                return
            # Only now — a real LLM turn — do we show the thinking spinner.
            self.show_spinner()
            try:
                await self._submit(line)
            finally:
                self.finalize_turn()
                # Refresh the status bar so session tok/s updates right after the turn.
                try:
                    self.query_one(StatusBar).refresh_status()
                except Exception:
                    pass

        # --- actions ------------------------------------------------------

        async def action_interrupt_or_quit(self) -> None:
            if self._turn_task and not self._turn_task.done():
                try:
                    if self.state.llm is not None:
                        self.state.llm._running = False
                except Exception:
                    pass
                self.log_system("⏹ interrupt requested", "yellow")
                return
            inp = self.query_one("#input", Input)
            if inp.value:
                inp.value = ""
                return
            self.exit()

        async def action_toggle_tool_output(self) -> None:
            self.state.show_tool_output = not self.state.show_tool_output
            self.log_system(f"tool output: {'on' if self.state.show_tool_output else 'off'}")

        async def action_toggle_thinking(self) -> None:
            self.state.show_thinking = not self.state.show_thinking
            self.log_system(f"thinking display: {'on' if self.state.show_thinking else 'off'}")

        async def action_thinking_mode(self) -> None:
            # Two-press confirm preserved via the slash handler
            await self._keyshortcut("thinking_toggle")

        async def action_pick_model(self) -> None:
            await self._keyshortcut("pick_model")

        # --- modal helpers -----------------------------------------------

        def show_help_modal(self) -> None:
            """Open the arrow-navigable /help modal. On selection, prefill
            the selected command into the input (with trailing space if the
            command takes args) and focus it."""

            def _on_close(choice):
                if not choice:
                    return
                try:
                    inp = self.query_one("#input", Input)
                except Exception:
                    return
                # Prefill with trailing space when the command takes args
                takes_args = False
                for name, hint, _desc in SLASH_COMMANDS:
                    if name == choice:
                        takes_args = bool(hint.strip())
                        break
                inp.value = choice + (" " if takes_args else "")
                try:
                    inp.cursor_position = len(inp.value)
                except Exception:
                    pass
                inp.focus()

            self.push_screen(HelpModal(), _on_close)

        async def confirm_tool(self, name: str, detail: str) -> bool:
            """Await a user approval via a Textual modal. Called from
            CLIState.confirm when `_TEXTUAL_APP` is active — replaces the
            blocking sys.stdin.readline() path that would deadlock the app.

            Implemented with a Future + callback rather than push_screen_wait,
            because the latter requires a Textual worker context, which the
            tool-confirmation call site (inside LLMService.chat) is not."""
            loop = asyncio.get_event_loop()
            fut: asyncio.Future = loop.create_future()

            def _cb(result) -> None:
                if not fut.done():
                    fut.set_result(bool(result))

            self.push_screen(ConfirmModal(name=name, detail=detail), _cb)
            try:
                return await fut
            except asyncio.CancelledError:
                return False

        def show_config_modal(self) -> None:
            """Open the unified /config modal. On apply, route the chosen
            model/vision/thinking/ctx through model_manager in a worker so
            the UI thread doesn't block on the ~3-7s llama-server reload."""
            current = self._current_config_snapshot()

            def _on_close(result):
                if not result:
                    return
                self.run_worker(self._apply_config(result), exclusive=True)

            self.push_screen(ConfigModal(current=current), _on_close)

        def _current_config_snapshot(self) -> dict:
            """Read current alias/ctx/vision/thinking from state + model_manager."""
            alias = getattr(self.state, "model_alias", None)
            try:
                from .services.model_manager import model_manager as _mm
                ctx = _mm.get_context_size()
                vision = _mm.vision_enabled
                thinking = _mm.thinking_enabled
            except Exception:
                ctx, vision, thinking = 0, True, False
            return {
                "alias": alias,
                "ctx": ctx or None,
                "vision": bool(vision),
                "thinking": bool(thinking),
            }

        async def _apply_config(self, cfg: dict) -> None:
            """Apply a ConfigModal result — switch model (if changed) and
            reload llama-server with the requested vision/thinking/ctx."""
            from .services.model_manager import (
                MODELS,
                MODEL_ALIASES,
                model_manager,
                resolve_model_id,
            )
            from .services import agents as agent_store

            alias = cfg.get("alias")
            ctx_val = cfg.get("ctx")
            vision = bool(cfg.get("vision", True))
            thinking = bool(cfg.get("thinking", False))

            if not alias:
                self.log_error("config: no model selected")
                return

            try:
                mid = resolve_model_id(alias)
            except Exception as e:
                self.log_error(f"config: {e}")
                return

            model_cfg = MODELS.get(mid, {})
            self.log_system(
                f"[cyan]applying config[/cyan] · model=[bold]{alias}[/bold] · "
                f"ctx={ctx_val or 'default'} · vision={'on' if vision else 'off'} · "
                f"thinking={'on' if thinking else 'off'}",
                style="dim",
            )

            if model_cfg.get("online"):
                # Online backends have no --mmproj / --reasoning knobs; just
                # re-point the client at the OpenRouter-compatible endpoint.
                import os as _os
                api_key = _os.getenv(model_cfg["api_key_env"], "")
                if not api_key:
                    self.log_error(f"missing {model_cfg['api_key_env']}")
                    return
                _os.environ["LLM_BASE_URL"] = model_cfg["base_url"]
                _os.environ["LLM_API_KEY"] = api_key
                _os.environ["LLM_MODEL"] = model_cfg["model_id"]
                self.state.model_alias = alias
                if self.state.llm is not None:
                    self.state.llm.switch_model(
                        _os.environ["LLM_BASE_URL"],
                        _os.environ["LLM_API_KEY"],
                        _os.environ["LLM_MODEL"],
                    )
                self.log_system(
                    f"[green]✓[/green] switched to [cyan]{self.state.model_label()}[/cyan]",
                    style="",
                )
                self._refresh_status_now()
                return

            # Local backend — reload llama-server with the new knobs
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model_manager.start(mid, vision, ctx_val, thinking),
                )
            except Exception as e:
                self.log_error(f"config reload failed: {e}")
                return

            # Persist preferences for the active agent
            try:
                patch = {
                    "model": alias,
                    "vision_enabled": vision,
                    "thinking_enabled": thinking,
                }
                if ctx_val is not None:
                    patch["ctx"] = int(ctx_val)
                agent_store.update_preferences(agent_store.get_active(), patch)
            except Exception:
                pass

            # Re-point the LLMService at the (possibly re-bound) llama endpoint
            import os as _os
            self.state.model_alias = alias
            if self.state.llm is not None:
                self.state.llm.switch_model(
                    _os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
                    _os.environ.get("LLM_API_KEY", "not-needed"),
                    _os.environ.get("LLM_MODEL", mid),
                )
            self.log_system(
                f"[green]✓[/green] config applied · [cyan]{self.state.model_label()}[/cyan]",
                style="",
            )
            self._refresh_status_now()

        def _refresh_status_now(self) -> None:
            try:
                self.query_one(StatusBar).refresh_status()
            except Exception:
                pass

        def on_mouse_down(self, event) -> None:
            """Capture the active text selection on RIGHT-click BEFORE
            Textual's default MouseUp handler clears it.

            Textual's Screen._on_mouse_down+MouseUp sequence calls
            clear_selection() when the down+up land on the same offset
            (regardless of button), so reading the selection from
            `on_click` was racy — usually empty by the time it ran. Here
            we read on MouseDown, set a flag so the Input's on_click
            knows not to paste, and let the default MouseUp still clear
            the highlight visually."""
            if getattr(event, "button", 0) != 3:
                return
            try:
                sel = self.screen.get_selected_text() or ""
            except Exception:
                sel = ""
            self._just_copied_selection = False
            if sel:
                try:
                    import pyperclip
                    pyperclip.copy(sel)
                    self._just_copied_selection = True
                    self.log_system(f"copied {len(sel)} chars to clipboard", "green")
                except Exception as e:
                    self.log_error(f"copy failed: {e}")

        def action_copy_chat(self) -> None:
            """Copy the whole chat transcript to the clipboard via OSC 52.

            Textual captures the mouse for its own events, so click-drag
            selection doesn't reach the terminal's native selection layer.
            (Shift-bypass of mouse capture is terminal-specific and fails on
            many setups.) F7 is the reliable keyboard path."""
            try:
                chat = self.query_one("#chat", VerticalScroll)
            except Exception:
                return
            parts: list[str] = []
            for child in chat.children:
                if isinstance(child, ChatMessage):
                    parts.append(child._buffer or "")
            transcript = "\n".join(p for p in parts if p).strip()
            if not transcript:
                self.log_system("nothing to copy")
                return
            try:
                self.copy_to_clipboard(transcript)
                n_lines = transcript.count("\n") + 1
                self.log_system(f"copied {n_lines} lines to clipboard", "green")
            except Exception as e:
                self.log_error(f"copy failed: {e}")

        def action_scroll_up(self) -> None:
            try:
                self.query_one("#chat", VerticalScroll).scroll_page_up()
            except Exception:
                pass

        def action_scroll_down(self) -> None:
            try:
                self.query_one("#chat", VerticalScroll).scroll_page_down()
            except Exception:
                pass

else:

    class AINowApp:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            raise RuntimeError("textual not installed — pip install textual")


__all__ = ["AINowApp", "HAS_TEXTUAL"]
