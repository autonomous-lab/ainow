"""Full-screen prompt_toolkit TUI for the AINow CLI.

Runs a long-lived `Application` that owns the whole terminal: scrollable chat
area on top, horizontal rule, input line, persistent two-line bottom toolbar.
The key win over `PromptSession` is that the toolbar stays visible during
streaming — tokens arrive into the chat area while the footer is always
painted at the bottom of the screen.

Falls back cleanly: if prompt_toolkit isn't available or the environment is
not a TTY, `cli.py` uses the existing line-based REPL instead.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Tuple

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.document import Document
    from prompt_toolkit.filters import Condition, has_focus
    from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import (
        ConditionalContainer,
        Float,
        FloatContainer,
        HSplit,
        Window,
    )
    from prompt_toolkit.layout.controls import (
        BufferControl,
        FormattedTextControl,
    )
    from prompt_toolkit.layout.dimension import Dimension as D
    from prompt_toolkit.layout.margins import ScrollbarMargin
    from prompt_toolkit.layout.processors import BeforeInput
    from prompt_toolkit.styles import Style
    _HAS_PTK = True
except ImportError:
    _HAS_PTK = False


HAS_FULLSCREEN = _HAS_PTK


# ──────────────────────────────────────────────────────────────────────────
# Chat log — append-only buffer of (style, text) fragments the chat window
# renders. Tokens, tool arrows, system messages all land here.
# ──────────────────────────────────────────────────────────────────────────

class ChatLog:
    """Append-only store of prompt_toolkit-style (style, text) fragments.

    Callers push fragments via `append`, `append_line`, `append_ansi`.
    Registered listeners (the Application) are notified on every change and
    invalidate the UI so the chat window repaints with the new content.
    """

    def __init__(self) -> None:
        self._fragments: List[Tuple[str, str]] = []
        self._listeners: List[Callable[[], None]] = []

    def on_change(self, cb: Callable[[], None]) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in self._listeners:
            try:
                cb()
            except Exception:
                pass

    def fragments(self) -> FormattedText:
        return FormattedText(self._fragments)

    def append(self, text: str, style: str = "") -> None:
        if not text:
            return
        self._fragments.append((style, text))
        self._notify()

    def append_line(self, text: str = "", style: str = "") -> None:
        """Append a fragment and ensure it terminates with a newline."""
        if not text.endswith("\n"):
            text = text + "\n"
        self.append(text, style)

    def append_ansi(self, ansi_text: str) -> None:
        """Parse an ANSI-coloured string (e.g. from Rich) into fragments."""
        if not ansi_text:
            return
        ft = to_formatted_text(ANSI(ansi_text))
        for style, text in ft:
            if text:
                self._fragments.append((style, text))
        self._notify()

    def clear(self) -> None:
        self._fragments = []
        self._notify()


# ──────────────────────────────────────────────────────────────────────────
# Path completer — identical to the PromptSession mode
# ──────────────────────────────────────────────────────────────────────────

if _HAS_PTK:
    import re as _re

    _FILE_REF_RX = _re.compile(r"@([A-Za-z0-9_./\\:-]*)$")

    class _FileRefCompleter(Completer):
        def __init__(self, cwd_getter: Callable[[], str]):
            self._cwd_getter = cwd_getter

        def get_completions(self, document: "Document", complete_event):
            text = document.text_before_cursor
            m = _FILE_REF_RX.search(text)
            if not m:
                return
            prefix = m.group(1).replace("\\", "/")
            start_position = -len(m.group(1))
            try:
                base = Path(self._cwd_getter()).resolve()
            except Exception:
                return
            if "/" in prefix:
                head, tail = prefix.rsplit("/", 1)
                target_dir = (base / head).resolve()
            else:
                tail = prefix
                target_dir = base
            try:
                if not str(target_dir).startswith(str(base)):
                    return
            except Exception:
                return
            if not target_dir.is_dir():
                return
            tail_low = tail.lower()
            try:
                entries = list(target_dir.iterdir())
            except PermissionError:
                return
            entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
            shown = 0
            for entry in entries:
                if shown >= 50:
                    break
                name = entry.name
                if name.startswith(".") and not tail.startswith("."):
                    continue
                if not name.lower().startswith(tail_low):
                    continue
                rel = entry.relative_to(base).as_posix()
                suffix = "/" if entry.is_dir() else ""
                yield Completion(
                    rel + suffix,
                    start_position=start_position,
                    display=name + suffix,
                    display_meta="dir" if entry.is_dir() else "",
                )
                shown += 1


# ──────────────────────────────────────────────────────────────────────────
# Shared git-branch helper (cached) for the toolbar
# ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────
# FullScreenApp
# ──────────────────────────────────────────────────────────────────────────

if _HAS_PTK:

    class FullScreenApp:
        """Run the REPL as a full-screen prompt_toolkit Application.

        Layout, top to bottom:
          * Chat history window (scrollable, grows to fill available rows)
          * Dim rule separator
          * Input line with `› ` prompt
          * Dim rule separator
          * Two-line toolbar (cwd + agent; ctx/perm/session + model)
        """

        def __init__(
            self,
            *,
            state,
            submit_handler: Callable[[str], Awaitable[None]],
            slash_handler: Callable[[str], Awaitable[bool]],
            bash_shortcut_handler: Callable[[str, bool], Awaitable[None]],
            keyshortcut_handler: Callable[[str], Awaitable[None]],
        ):
            self.state = state
            self._submit = submit_handler
            self._slash = slash_handler
            self._bash = bash_shortcut_handler
            self._keyshortcut = keyshortcut_handler
            self.log = ChatLog()

            self._turn_task: Optional[asyncio.Task] = None
            self._turn_running = asyncio.Event()

            # Input buffer
            histfile = str(Path(os.path.expanduser("~")) / ".ainow_cli_history")
            self.input_buffer = Buffer(
                multiline=False,
                history=FileHistory(histfile),
                accept_handler=self._accept,
                completer=_FileRefCompleter(lambda: state.llm._cwd if state.llm else os.getcwd()),
                complete_while_typing=True,
            )

            # Rebind ChatLog notifications to redraw AND auto-scroll to bottom.
            def _on_log_change():
                # Snap back to the bottom on any new content so the user
                # always sees the latest token, unless they've explicitly
                # scrolled up (we don't distinguish "user scroll" from
                # "new content" currently — simpler to always snap).
                self._scroll_offset_lines = 0
                self._invalidate()
            self.log.on_change(_on_log_change)

            # Build layout + app
            self.app = self._build_app()

        # --- construction helpers -----------------------------------------------

        def _build_app(self) -> "Application":
            from prompt_toolkit.layout.dimension import Dimension

            # Manual scroll state — how many lines from the bottom we are.
            # 0 = pinned to the latest message. Positive = scrolled up N lines.
            self._scroll_offset_lines = 0

            chat_control = FormattedTextControl(
                self.log.fragments,
                show_cursor=False,
                focusable=False,
            )

            # Window-level vertical_scroll getter: computes the scroll index so
            # the tail of the log is visible by default, and PgUp/PgDn shift
            # it by `_scroll_offset_lines`. No cursor-position hacks.
            def _get_vscroll(window):
                total = sum(t.count("\n") for _, t in self.log._fragments)
                # Best-effort window height (1 row fallback during early paint).
                try:
                    height = window.render_info.window_height if window.render_info else 20
                except Exception:
                    height = 20
                # Pin to bottom, then subtract user's scroll offset.
                scroll = max(0, total - height + 1)
                scroll = max(0, scroll - self._scroll_offset_lines)
                return scroll

            # Force the chat window to be fully flex — `height=D(weight=1)`
            # tells HSplit "take all leftover screen space". Without this the
            # Window's preferred_height equals its content line count, so HSplit
            # grows the chat window by one row per streamed line and pushes
            # the input line progressively down the screen.
            # Chat fills the whole screen. The footer (rules + input +
            # toolbar) overlays the bottom 5 rows via a Float — that keeps
            # the input anchored at a FIXED absolute position regardless of
            # what happens in the chat window's layout.
            chat_window = Window(
                content=chat_control,
                wrap_lines=True,
                right_margins=[ScrollbarMargin(display_arrows=True)],
                always_hide_cursor=True,
                get_vertical_scroll=_get_vscroll,
                # Huge `preferred` forces HSplit to cap us at the actual
                # remaining rows — effectively "fill everything the footer
                # doesn't need". Because `preferred_specified=True`, this
                # overrides the default preferred_height (= content lines)
                # and kills the turn-by-turn layout drift.
                height=D(min=3, preferred=99999, weight=1),
            )
            self._chat_window = chat_window

            # Two separate rule Windows — sharing a single Window instance at
            # two positions in HSplit makes prompt_toolkit re-render it twice
            # and drift the layout down by one row per turn.
            rule_top = Window(height=1, char="─", style="fg:ansibrightblack")
            rule_bot = Window(height=1, char="─", style="fg:ansibrightblack")

            input_window = Window(
                content=BufferControl(
                    buffer=self.input_buffer,
                    input_processors=[BeforeInput("› ", style="fg:ansimagenta bold")],
                    focusable=True,
                ),
                # Fixed 1-line height. With Dimension(max=6) the HSplit's flex
                # arithmetic grew the input by one row every time the chat
                # window claimed extra content lines, pushing the prompt down
                # screen after each turn. One row is strict and predictable;
                # multi-line paste is still usable — prompt_toolkit wraps it
                # visually inside the single line (scrolls horizontally).
                height=1,
                style="fg:ansicyan bold",
            )

            toolbar_window = Window(
                content=FormattedTextControl(
                    self._toolbar_fragments,
                    show_cursor=False,
                    focusable=False,
                ),
                # Strict 2-row footer. `wrap_lines=False` prevents line 2 from
                # wrapping onto a ghost 3rd row when the content is just a
                # couple of chars too wide — that ghost row was eating one row
                # from the chat window and shifting the input one line down
                # per turn (once session stats turned a medium-length line
                # into a long one).
                height=D.exact(2),
                wrap_lines=False,
                dont_extend_height=True,
            )

            # Group the footer (rules + input + toolbar) so it has a
            # predictable total height regardless of its sibling's content.
            # Total height = 1 (rule_top) + 1 (input) + 1 (rule_bot) + 2
            # (toolbar) = 5 rows, exact.
            footer = HSplit(
                [rule_top, input_window, rule_bot, toolbar_window],
                height=D.exact(5),
            )

            root = HSplit([
                chat_window,  # grabs all remaining rows
                footer,
            ])

            layout = Layout(root, focused_element=input_window)

            style = Style.from_dict({
                # Default body text — override elsewhere per-fragment.
                "": "",
            })

            return Application(
                layout=layout,
                key_bindings=self._build_keybindings(),
                style=style,
                full_screen=True,
                # Mouse support off — when prompt_toolkit captures mouse it
                # also consumes drag events, which breaks terminal-native
                # text selection (users have to hold Shift). Keep the OS
                # selection behavior; we don't have clickable widgets anyway.
                mouse_support=False,
                # Don't let prompt_toolkit's built-in page-navigation bindings
                # steal our custom PgUp / PgDown handlers (they're intended for
                # completion menus and end up shadowing chat scroll).
                enable_page_navigation_bindings=False,
            )

        def _build_keybindings(self) -> "KeyBindings":
            kb = KeyBindings()

            @kb.add("c-c")
            def _ctrl_c(event):
                # Ctrl+C semantics:
                #   - running turn → ask the LLM to stop cleanly
                #   - input has text → clear it
                #   - input is empty → exit the app (same as Ctrl+D)
                if self._turn_task and not self._turn_task.done():
                    try:
                        if self.state.llm is not None:
                            self.state.llm._running = False
                    except Exception:
                        pass
                    self.log.append_line("⏹ interrupt requested", "fg:ansiyellow")
                    return
                if self.input_buffer.text:
                    self.input_buffer.reset()
                    return
                event.app.exit(result=None)

            @kb.add("c-d")
            def _ctrl_d(event):
                event.app.exit(result=None)

            @kb.add("c-o")
            def _ctrl_o(event):
                self.state.show_tool_output = not self.state.show_tool_output
                self._invalidate()

            @kb.add("c-t")
            def _ctrl_t(event):
                self.state.show_thinking = not self.state.show_thinking
                self._invalidate()

            @kb.add("s-tab")
            def _shift_tab(event):
                asyncio.get_event_loop().create_task(self._keyshortcut("thinking_toggle"))
                self._invalidate()

            @kb.add("c-l")
            def _ctrl_l(event):
                asyncio.get_event_loop().create_task(self._keyshortcut("pick_model"))

            # --- Scroll bindings for the chat window ------------------------
            # Chat is focused-at-bottom by default (offset=0). Users can
            # scroll up via PageUp / Alt+Up / Ctrl+Up to review history.
            # Any new content snaps back to the bottom by design — we reset
            # the offset in `append` via _notify → invalidate.

            def _scroll_up(lines: int):
                self._scroll_offset_lines += lines
                self._invalidate()

            def _scroll_down(lines: int):
                self._scroll_offset_lines = max(0, self._scroll_offset_lines - lines)
                self._invalidate()

            @kb.add("pageup")
            def _pg_up(event):
                try:
                    rows = max(3, event.app.output.get_size().rows - 8)
                except Exception:
                    rows = 10
                _scroll_up(rows)

            @kb.add("pagedown")
            def _pg_dn(event):
                try:
                    rows = max(3, event.app.output.get_size().rows - 8)
                except Exception:
                    rows = 10
                _scroll_down(rows)

            @kb.add("escape", "up")         # Alt+Up
            def _alt_up(event):
                _scroll_up(1)

            @kb.add("escape", "down")       # Alt+Down
            def _alt_down(event):
                _scroll_down(1)

            @kb.add("escape", "end")        # Alt+End → jump to bottom
            def _alt_end(event):
                self._scroll_offset_lines = 0
                self._invalidate()

            return kb

        # --- input handling ---------------------------------------------------

        def _accept(self, buffer: "Buffer") -> bool:
            text = buffer.text
            buffer.reset()
            if not text.strip():
                return True
            # Dispatch the turn in the background so the input line reopens
            # immediately. The ChatLog gets the output as tokens arrive.
            if self._turn_task and not self._turn_task.done():
                self.log.append_line("(a turn is already running — press Ctrl+C to interrupt)", "fg:ansiyellow")
                return True
            self._turn_task = asyncio.get_event_loop().create_task(self._handle_line(text))
            return True

        async def _handle_line(self, line: str) -> None:
            line = line.strip()
            if not line:
                return
            # Echo the user's input into the chat window.
            self.log.append_line(f"› {line}", "fg:ansimagenta bold")

            # ! / !! shortcut — shell passthrough
            if line.startswith("!!"):
                await self._bash(line[2:].strip(), True)
                return
            if line.startswith("!") and not line.startswith("!/"):
                await self._bash(line[1:].strip(), False)
                return

            # Slash commands
            try:
                handled = await self._slash(line)
            except SystemExit:
                self.app.exit(result=None)
                return
            if handled:
                return

            # Normal LLM turn
            self._turn_running.set()
            try:
                await self._submit(line)
            finally:
                self._turn_running.clear()

        # --- toolbar ---------------------------------------------------------

        def _toolbar_fragments(self) -> FormattedText:
            s = self.state
            used, mx = s.context_stats()
            pct = int(used / mx * 100) if mx else 0
            pct_color = "fg:ansigreen" if pct < 70 else "fg:ansiyellow" if pct < 85 else "fg:ansired"
            perm = "yolo" if s.yolo else "confirm"
            perm_color = "fg:ansiyellow" if s.yolo else "fg:ansigreen"

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

            DIM = "fg:ansibrightblack"
            CYAN = "fg:ansicyan"
            WHITE = "fg:ansiwhite"

            model_name = s.model_label()
            if len(model_name) > 40:
                model_name = model_name[:39] + "…"

            try:
                from .services.model_manager import model_manager as _mm
                reasoning = "high" if _mm.thinking_enabled else "off"
            except Exception:
                reasoning = "off"

            mx_fmt = (f"{mx // 1024}K" if mx and mx >= 1024 else f"{mx}") if mx else "?"

            try:
                cols = self.app.output.get_size().columns
            except Exception:
                import shutil as _sh
                cols = _sh.get_terminal_size(fallback=(100, 40)).columns
            # Subtract the scrollbar margin + a tiny safety buffer so we never
            # render a line that's one or two chars too wide (which was
            # wrapping into a ghost 3rd row and shifting the layout).
            cols = max(40, cols - 2)

            line1_parts = [(CYAN, short)]
            if branch:
                line1_parts += [(DIM, " ("), (WHITE, branch), (DIM, ")")]
            line1_parts += [(DIM, "  ·  agent "), (CYAN, s.agent_name)]

            left_parts: List[Tuple[str, str]] = [
                (DIM, "ctx "),
                (pct_color, f"{pct}%/{mx_fmt}"),
                (DIM, "  ·  "),
                (perm_color, perm),
            ]

            right_parts: List[Tuple[str, str]] = [
                (CYAN + " bold", model_name),
                (DIM, "  ·  "),
                (WHITE, reasoning),
            ]

            # Session stats are optional — we add them only if there's room so
            # the toolbar stays single-line (ansi wrap would eat the input).
            sess_tok = getattr(s, "session_total_tokens", 0)
            sess_sec = getattr(s, "session_total_seconds", 0.0)
            session_extra: List[Tuple[str, str]] = []
            if sess_tok > 0 and sess_sec > 0:
                avg_tps = sess_tok / sess_sec
                sess_fmt = f"{sess_tok}" if sess_tok < 1000 else f"{sess_tok // 1000}.{(sess_tok % 1000) // 100}k"
                session_extra = [
                    (DIM, "  ·  session "),
                    (WHITE, f"{sess_fmt} tok"),
                    (DIM, " @ "),
                    (WHITE, f"{avg_tps:.1f} tok/s"),
                ]

            def _total_len(parts):
                return sum(len(t) for _, t in parts)

            base_len = _total_len(left_parts) + _total_len(right_parts)
            session_len = _total_len(session_extra)

            # Always reserve 2 spaces of padding between left and right.
            if session_len and base_len + session_len + 2 <= cols:
                left_parts = left_parts + session_extra

            left_len = _total_len(left_parts)
            right_len = _total_len(right_parts)
            pad = max(2, cols - left_len - right_len)
            line2_parts = left_parts + [("", " " * pad)] + right_parts

            # Truncate line 1 hard (cwd + branch + agent) if needed.
            line1_text = "".join(t for _, t in line1_parts)
            if len(line1_text) > cols:
                # Crude fallback: just show the cwd basename.
                line1_parts = [(CYAN, Path(short).name or short), (DIM, "  ·  agent "), (CYAN, s.agent_name)]

            return FormattedText(line1_parts + [("", "\n")] + line2_parts)

        # --- plumbing -------------------------------------------------------

        def _invalidate(self) -> None:
            try:
                self.app.invalidate()
            except Exception:
                pass

        def _emit_banner(self) -> None:
            """Paint the startup banner into the chat window so it stays
            visible inside the full-screen app (the Rich Console banner was
            getting wiped when the Application took over the terminal)."""
            s = self.state
            model_label = s.model_label()
            backend = s.backend_label()
            perm = "yolo" if s.yolo else "confirm"
            perm_color = "fg:ansiyellow" if s.yolo else "fg:ansigreen"
            try:
                from .services.model_manager import model_manager as _mm
                ctx = _mm.get_context_size()
            except Exception:
                ctx = 0
            ctx_fmt = (f"{ctx // 1024}K" if ctx and ctx >= 1024 else f"{ctx}") if ctx else "unknown"

            DIM = "fg:ansibrightblack"
            MAG = "fg:ansimagenta bold"
            CYAN = "fg:ansicyan"

            frags = [
                (MAG, "AINow"), (DIM, "  ·  "), (DIM, "v1.1"), ("", "\n"),
                (DIM, "Local-first AI agent framework — chat, code, voice, vision"), ("", "\n"),
                ("", "\n"),
                (DIM, "model:       "), (CYAN, f"{model_label}  ({backend})"), ("", "\n"),
                (DIM, "agent:       "), (CYAN, s.agent_name), ("", "\n"),
                (DIM, "permissions: "), (perm_color, perm), ("", "\n"),
                (DIM, "context:     "), (CYAN, ctx_fmt), ("", "\n"),
                ("", "\n"),
                (DIM, "/help for commands  ·  /model to switch  ·  Ctrl+D to exit"), ("", "\n"),
                ("", "\n"),
            ]
            for style, text in frags:
                self.log._fragments.append((style, text))

        async def run(self) -> None:
            self._emit_banner()
            # Force the alternate screen buffer explicitly.
            # `Application(full_screen=True)` is supposed to do this, but on
            # Windows Terminal / conhost we were seeing successive frames
            # rendered ON TOP of each other (toolbar lines accumulating,
            # input vanishing) — strong sign that the enter-altscreen escape
            # isn't reaching the terminal. Emit it ourselves, then restore on
            # exit so the user's scrollback is preserved.
            _altscreen_active = False
            try:
                if sys.stdout.isatty():
                    sys.stdout.write("\033[?1049h\033[2J\033[H")
                    sys.stdout.flush()
                    _altscreen_active = True
            except Exception:
                pass
            try:
                await self.app.run_async()
            finally:
                if _altscreen_active:
                    try:
                        sys.stdout.write("\033[?1049l")
                        sys.stdout.flush()
                    except Exception:
                        pass

else:

    class FullScreenApp:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            raise RuntimeError("prompt_toolkit not installed — full-screen TUI unavailable")


__all__ = ["FullScreenApp", "ChatLog", "HAS_FULLSCREEN"]
