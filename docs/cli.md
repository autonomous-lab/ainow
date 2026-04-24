# CLI (headless mode)

Use AINow as a local coding / chat agent without the browser — a full-screen Textual TUI by default (or line-based REPL with `--no-textual`) with slash commands, a persistent status bar, and `@path` file expansion.

![AINow CLI](cli-screenshot.png)

```bash
# Start the interactive TUI (attaches to a running llama-server if the web UI is up)
python -m src.cli -i

# One-shot
python -m src.cli "list all python files under src/ and tell me the biggest one"

# Pick agent + model
python -m src.cli -a donald-trump -m 27b-iq2 "write me a haiku"

# Auto-approve every tool call (use with care)
python -m src.cli --yolo "clean up the dead imports in src/services/llm.py"

# Line-based REPL instead of the Textual TUI
python -m src.cli -i --no-textual

# One-line banner instead of the boxed panel
python -m src.cli -i --minimal-banner

# Take over the terminal (alt-screen, à la vim / less). Default is off so
# you keep terminal scrollback; opt-in if you want a "full TUI" feel.
python -m src.cli -i --altscreen
```

## Fast cold start

On launch the CLI probes `http://127.0.0.1:8080/health` + `/v1/models`; if a llama-server is already running with the requested model (typically because the web UI is open), the CLI **attaches** to it instead of restarting. llama-server cold start is 3–7s, attach is <200ms. The attach reads `/props` for the actual running context size, so if the web UI started llama-server with a 128K override, the CLI banner reflects that.

If the web UI is closed the CLI falls back to the same auto-download / auto-build path `main.py` uses. LLMService imports are deferred to the first turn to keep `--help` and the banner sub-second.

## REPL slash commands

| Command | What it does |
|---|---|
| `/help` | arrow-navigable list of all commands (Enter prefills into input) |
| `/config` | one-shot modal: model + vision + thinking + ctx, applies in a single llama-server reload |
| `/model [alias]` | switch model (e.g. `/model 27b-iq2`, `/model online`) or list all with active marker |
| `/agent` | list all agents with active marker, model, lang, MCP + task counts |
| `/agent <name>` | switch to an agent |
| `/agent new <name>` | create a new agent |
| `/agent delete <name>` | delete an agent (confirm required even under `--yolo`) |
| `/agent edit [name]` | open `CLAUDE.md` in `$EDITOR` — re-read on next turn |
| `/agent info [name]` | full agent detail (cwd, model, lang, voice, vision, MCP list, …) |
| `/thinking` | toggle reasoning mode on/off (reloads llama-server) |
| `/permissions [mode]` | switch between `yolo` and `confirm` at runtime |
| `/context` | detailed context breakdown (used / max, msg count) |
| `/history` | print conversation history with message indices |
| `/clear` | reset conversation history (keeps agent + model) |
| `/compact` | force context compaction now |
| `/save [id]` | save current session (auto-generated id if omitted) |
| `/load <id>` | load a named session |
| `/tree` | list saved sessions |
| `/fork <idx>` | branch a new session from message index (see `/history`) |
| `/skills` | list loaded skill-knowledge packs with their triggers |
| `/cwd` | show the tool working directory |
| `/verbose` | toggle live thinking-token display |
| `/quit` \| `/exit` | leave |

Slash-command autocomplete is live in the Textual TUI — start typing `/` and the completer suggests the matching command. Tab accepts the suggestion.

## Inline shortcuts

| Syntax | What it does |
|---|---|
| `!cmd` | run `cmd` in the shell, show output (no LLM call) |
| `!!cmd` | run `cmd` silently (output not displayed) |
| `@path/to/file` | inline-expand the file contents into the prompt (sandboxed to the agent's cwd) |

Absolute image paths pasted into the prompt (`C:\…\pic.jpg`, `/home/user/photo.png`, quoted or bare) are **auto-attached as vision input** — no need to drag/drop.

## Visual feedback

- **Startup banner** — model, backend (`llama.cpp` or `cloud`), permissions, context window, vision/thinking status, active agent. `--minimal-banner` for a one-line version.
- **Persistent status bar** (Textual) — cwd + git branch + agent, ctx usage %, session tok/s running total.
- **Thinking spinner** — animated braille spinner between submit and the first LLM token ("AINow is thinking…", "warming up the KV cache…", etc.).
- **Thinking bubble** — when `/verbose` or Ctrl+T is on, reasoning_content streams live into a dedicated yellow-italic "💭 thinking…" bubble with a "— thought for Xs —" footer.
- **Tool-use flow** — `→ bash  git status` followed by `✓ → bash  2 lines (52 chars)`. Errors show as `✗`. Ctrl+O expands each tool result inline.
- **Confirm modal** — `? confirm bash  rm -rf /tmp/foo` — press Enter/y to approve, Esc/n to deny. Skipped entirely under `--yolo`.

## Keyboard shortcuts (Textual TUI)

| Key | Action |
|---|---|
| `Ctrl+O` | toggle full tool-output expansion |
| `Ctrl+T` | toggle live thinking-token display |
| `Shift+Tab` | toggle reasoning mode — **two-press confirm** (5s window) to avoid an accidental llama-server reload |
| `Ctrl+L` | quick model picker |
| `Ctrl+C` during a turn | first press signals graceful stop, second aborts hard |
| `↑` / `↓` | history navigation (persisted in `~/.ainow_cli_history`) |
| `Tab` | accept autocomplete (`/` command or `@path/to/file`) |
| `PageUp` / `PageDown` | scroll the chat area |
| `F7` | copy the full chat transcript to the clipboard (OSC 52) |
| `Right-click` on chat | copy the current text selection |
| `Right-click` on input | paste clipboard into input (no selection) |
| `Ctrl+V` | paste clipboard (multi-line flattened to spaces) |
| `Ctrl+D` | exit |

## Notes

The CLI reuses the same `LLMService`, tool registry, skill-knowledge packs, and agent workspace as the web UI — so your `CLAUDE.md` persona, MCP servers, tool permissions, and session history all apply. `llama-server` is started via the same auto-download / auto-build path `main.py` uses.

Tokens stream to **stdout** so you can pipe them (`python -m src.cli "…" | tee out.txt`); Rich status lines, tool arrows, and confirm prompts go to **stderr**, so piping stdout stays clean.
