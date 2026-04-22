"""Prompt-toolkit TUI layer for AINow's CLI.

Adds over the plain readline REPL:
  * Persistent bottom toolbar showing model / agent / ctx% / cwd / key hints
  * Key bindings:
      - Ctrl+O: toggle full tool-output display
      - Ctrl+T: toggle thinking-token display
      - Shift+Tab: toggle thinking mode (reloads llama-server)
      - Ctrl+L: quick model picker
      - Ctrl+R: reverse incremental search (built-in)
      - Ctrl+A / Ctrl+E / Ctrl+W / Ctrl+U: line edit (built-in)
  * `@path/to/file` path-completer with fuzzy tail match, sandboxed to agent cwd
  * Persistent history across sessions in ~/.ainow_cli_history

Gracefully falls back to the plain readline REPL if prompt_toolkit is missing.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional


# Cache git-branch lookups so the toolbar doesn't hit the filesystem on every
# keystroke. Invalidate after 5s.
_BRANCH_CACHE: dict[str, tuple[float, str]] = {}
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

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.styles import Style
    _HAS_PTK = True
except ImportError:
    _HAS_PTK = False


HAS_TUI = _HAS_PTK


_FILE_REF_RX = re.compile(r"@([A-Za-z0-9_./\\:-]*)$")


def _html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if _HAS_PTK:

    class _FileRefCompleter(Completer):
        """Completes `@path/to/file` tokens, sandboxed to the agent's cwd.

        Triggered whenever the token preceding the cursor starts with `@`.
        Case-insensitive prefix match on the last path segment.
        """

        def __init__(self, cwd_getter: Callable[[], str]):
            self._cwd_getter = cwd_getter

        def get_completions(self, document, complete_event):
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
                head = ""
                tail = prefix
                target_dir = base
            # Stay within the sandbox
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
            # Dirs first, alpha-sorted, limit to 50 suggestions
            entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
            shown = 0
            for entry in entries:
                if shown >= 50:
                    break
                name = entry.name
                if name.startswith("."):
                    # hide dotfiles unless explicitly requested
                    if not tail.startswith("."):
                        continue
                if not name.lower().startswith(tail_low):
                    continue
                rel = entry.relative_to(base).as_posix()
                suffix = "/" if entry.is_dir() else ""
                yield Completion(
                    rel + suffix,
                    start_position=start_position,
                    display=name + suffix,
                    display_meta="dir" if entry.is_dir() else f"{entry.stat().st_size}b",
                )
                shown += 1

    class TUIPrompt:
        """prompt_toolkit-backed input + toolbar, wired to a CLIState."""

        def __init__(
            self,
            *,
            state,
            on_shortcut: Callable[[str], Awaitable[None]],
        ):
            self.state = state
            self._on_shortcut = on_shortcut
            self._histfile = str(Path(os.path.expanduser("~")) / ".ainow_cli_history")
            self._kb = self._build_keybindings()

            # Kill the default reverse-video bg on the toolbar (which was
            # making each colored segment look like a highlighted block).
            # Colorize the prompt char and user input so they stand out in the
            # scrollback against assistant output.
            self._style = Style.from_dict({
                "bottom-toolbar": "noreverse bg:default fg:default",
                "bottom-toolbar.text": "bg:default",
                "promptchar": "fg:ansimagenta bold",
                # The empty-string class styles all unclassified text — in a
                # PromptSession that's exactly the input the user types.
                "": "fg:ansicyan bold",
            })

            self._session = PromptSession(
                message=[("class:promptchar", "› ")],
                history=FileHistory(self._histfile),
                bottom_toolbar=self._bottom_toolbar,
                key_bindings=self._kb,
                completer=_FileRefCompleter(lambda: self.state.llm._cwd if self.state.llm else os.getcwd()),
                complete_while_typing=True,
                complete_style=CompleteStyle.MULTI_COLUMN,
                multiline=False,
                enable_suspend=True,
                style=self._style,
            )

        def _bottom_toolbar(self):
            """pi.dev-inspired footer.

            Three lines, minimal chrome:
              1. horizontal rule under the input
              2. cwd (branch) · agent
              3. ctx XX%/MAX · permissions          [right-aligned] model · reason
            Keyboard hints stay out of the footer — /help lists them.
            """
            import shutil
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

            # Try to read terminal width from prompt_toolkit's app so the
            # right-alignment lines up even after resize.
            try:
                from prompt_toolkit.application.current import get_app
                cols = get_app().output.get_size().columns
            except Exception:
                cols = shutil.get_terminal_size(fallback=(100, 40)).columns

            rule = [(DIM, "─" * max(0, cols))]

            # Line 2: cwd (branch) · agent
            line2_parts = [(CYAN, short)]
            if branch:
                line2_parts += [(DIM, " ("), (WHITE, branch), (DIM, ")")]
            line2_parts += [(DIM, "  ·  agent "), (CYAN, s.agent_name)]

            # Line 3: ctx + perm on the left, model · reason right-aligned
            left_parts = [
                (DIM, "ctx "),
                (pct_color, f"{pct}%/{mx_fmt}"),
                (DIM, "  ·  "),
                (perm_color, perm),
            ]
            # Session-wide token stats: total tokens streamed, average tok/s.
            sess_tok = getattr(s, "session_total_tokens", 0)
            sess_sec = getattr(s, "session_total_seconds", 0.0)
            if sess_tok > 0 and sess_sec > 0:
                avg_tps = sess_tok / sess_sec
                sess_fmt = f"{sess_tok}" if sess_tok < 1000 else f"{sess_tok // 1000}.{(sess_tok % 1000) // 100}k"
                left_parts += [
                    (DIM, "  ·  session "),
                    (WHITE, f"{sess_fmt} tok"),
                    (DIM, " @ "),
                    (WHITE, f"{avg_tps:.1f} tok/s"),
                ]
            right_parts = [
                (CYAN + " bold", model_name),
                (DIM, "  ·  "),
                (WHITE, reasoning),
            ]
            # Measure text widths for the pad
            left_len = sum(len(t) for _, t in left_parts)
            right_len = sum(len(t) for _, t in right_parts)
            pad = max(2, cols - left_len - right_len)
            line3 = left_parts + [("", " " * pad)] + right_parts

            return FormattedText(
                rule
                + [("", "\n")] + line2_parts
                + [("", "\n")] + line3
            )

        def _build_keybindings(self) -> "KeyBindings":
            kb = KeyBindings()

            @kb.add("c-o")
            def _toggle_tool_out(event):
                self.state.show_tool_output = not self.state.show_tool_output
                event.app.invalidate()

            @kb.add("c-t")
            def _toggle_thinking_view(event):
                self.state.show_thinking = not self.state.show_thinking
                event.app.invalidate()

            @kb.add("s-tab")
            def _cycle_thinking(event):
                # Async: toggle thinking mode (reloads llama-server).
                # We can't await inside the handler, so schedule on the loop.
                loop = asyncio.get_event_loop()
                loop.create_task(self._on_shortcut("thinking_toggle"))
                event.app.invalidate()

            @kb.add("c-l")
            def _pick_model(event):
                loop = asyncio.get_event_loop()
                loop.create_task(self._on_shortcut("pick_model"))

            return kb

        async def prompt_async(self) -> Optional[str]:
            # Print the rule above the input area — pairs with the rule below
            # drawn by the bottom_toolbar to give the input a framed look.
            try:
                import shutil
                from prompt_toolkit.application.current import get_app
                try:
                    cols = get_app().output.get_size().columns
                except Exception:
                    cols = shutil.get_terminal_size(fallback=(100, 40)).columns
                # ANSI dim rule on stderr so it doesn't pollute stdout piping
                import sys as _sys
                _sys.stderr.write("\033[90m" + ("─" * max(0, cols)) + "\033[0m\n")
                _sys.stderr.flush()
            except Exception:
                pass
            try:
                line = await self._session.prompt_async()
                return line
            except (EOFError, KeyboardInterrupt):
                return None

else:

    class TUIPrompt:  # type: ignore[no-redef]
        """Stub — prompt_toolkit not installed. Callers should check HAS_TUI."""

        def __init__(self, **kwargs):
            raise RuntimeError("prompt_toolkit not installed")

        async def prompt_async(self) -> Optional[str]:
            return None


__all__ = ["TUIPrompt", "HAS_TUI"]
