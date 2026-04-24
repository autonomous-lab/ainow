# Agents, MCP, Scheduled Tasks, Tools

## Agents

Each agent is a self-contained workspace at `agents/<name>/`:

```
agents/
  default/
    CLAUDE.md              # System prompt / persona
    meta.json              # Display name + mcp_servers config
    sessions/              # Conversation history (auto-saved)
    scheduled_tasks.json   # Cron AI tasks for this agent
    profile.jpg            # Optional avatar shown in the sidebar
  donald-trump/
    CLAUDE.md
    meta.json
    sessions/
    .skills/               # User-authored bash scripts (optional)
    memory.db              # User skill data (optional)
    profile.jpg
```

Switch agents from the **agent picker** in the sidebar — clicking swaps the active CLAUDE.md, sessions list, MCP servers, scheduled tasks, tool cwd, and avatar all at once. The currently-selected agent's `cwd` is its own folder, so file/bash/grep tools resolve relative paths under `agents/<name>/`.

UI affordances per agent:

- **Agent dropdown** with `+` (new agent) and `×` (delete) buttons
- **Profile image**: drop a `profile.jpg` / `.png` / `.webp` in the agent folder and it appears under the picker
- **Edit Profile** button → opens a large modal editor for the agent's `CLAUDE.md` with `Cmd/Ctrl+S` to save
- **MCP Servers** button → opens the MCP servers modal (see below)
- **Scheduled Tasks** button → opens the cron AI modal (see below)
- **Export Agent** → downloads the agent as a `.tar.gz` archive (everything except sessions/node_modules/caches)
- **Import Agent** → uploads a `.tar.gz` archive, extracts into `agents/<name>/`, auto-patches known skills for cross-platform compatibility (e.g. memory-db → sqlite-vec)

A migration is run automatically the first time you start: any legacy `~/.ainow/sessions` or `~/.ainow/agents` content gets moved into `<cwd>/agents/default/` so your old conversations carry over.

The `default` agent is auto-created on first run and cannot be deleted.

## MCP (Model Context Protocol)

Each agent can declare MCP servers in its `meta.json`:

```json
{
  "display_name": "donald-trump",
  "mcp_servers": {
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..." }
    }
  }
}
```

When the agent activates:

1. The previous agent's MCP servers are torn down (subprocess killed, tools unregistered)
2. The new agent's servers are spawned via stdio in dedicated long-lived asyncio tasks
3. Each server's `tools/list` is queried and every tool is registered into AINow's `TOOL_REGISTRY` under a namespaced name `mcp__<server>__<tool>` so collisions are impossible
4. The LLM sees them in its tool schema on the next turn (schemas are re-fetched per request, so newly-loaded tools appear immediately)
5. The sidebar shows a chip per loaded server with its tool count

MCP tools require user confirmation by default (the protocol has no standard `read_only` flag and we can't trust third-party servers blindly).

### MCP Servers UI

Click **MCP Servers** under the agent picker. The modal lets you list/add/edit/delete servers, with a **Suggested · click to add** section grouping 24 well-known servers across categories:

- **Reference**: Everything (test), Filesystem, Memory (knowledge graph), Sequential Thinking, Time
- **Web & Search**: Fetch, Tavily 🔑, Exa 🔑, Firecrawl 🔑, Playwright
- **Dev Tools**: Git, GitHub 🔑, GitLab 🔑, Sentry 🔑
- **Databases**: SQLite, Postgres, Redis
- **Productivity**: Notion 🔑, Linear 🔑, Obsidian 🔑
- **Communication**: Slack 🔑
- **Cloud**: Cloudflare, AWS Bedrock KB 🔑
- **Media**: YouTube Transcript

🔑 chip = needs an API key; clicking the suggestion auto-opens the form with the env field ready. Servers run via `npx -y` (TypeScript packages) or `uvx` (Python packages — make sure `uv` is on PATH).

The MCP ecosystem has 5,400+ servers in the official registry; the suggested list is a curated starting point.

## Scheduled Tasks (Cron AI)

Per-agent recurring or one-time prompts that fire on a schedule. Each task runs your prompt either:

- **As a new session** — headless agent execution, result saved as a chat in the sidebar (you see it next time you open the conversation list)
- **Injected into the active chat** — falls back to a new session if no live conversation is open for that agent

Tasks live at `agents/<name>/scheduled_tasks.json`:

```json
{
  "tasks": [
    {
      "id": "task_...",
      "name": "Daily news digest",
      "schedule": "0 9 * * *",
      "prompt": "Use Tavily to search for today's top news on Iran and summarise.",
      "mode": "new_session",
      "enabled": true,
      "run_count": 7,
      "last_run": "...",
      "last_status": "ok"
    }
  ]
}
```

Schedules accept cron expressions (`MIN HOUR DAY MONTH WEEKDAY`) or one-time ISO dates (`YYYY-MM-DD HH:MM`). One-time tasks are auto-disabled after they fire successfully.

### Scheduled Tasks UI

Click **Scheduled Tasks** under the agent picker. The modal lets you:

- See each task with its schedule, mode badge, status badge (`ok` / `error` / `disabled`), run count, and last-run timestamp
- **▶ Run** any task immediately (great for testing)
- Toggle tasks on/off, edit, delete
- Add a new task with:
  - **Schedule preset dropdown** (Every minute / Every 5 min / Every hour / Every day at 9 AM / Every weekday at 9 AM / Every Monday / etc.)
  - **Live "next 3 runs" preview** that recomputes as you type — invalid expressions show in red
  - **Mode radio** to pick new-session vs inject-into-active

When a task fires while you're connected, you get a toast notification (`📅 <task name> ran`) and the conversation list auto-refreshes to show any new sessions.

Cron times use the **server's local time** (not UTC). Headless task execution auto-approves any tool call (no user in the loop).

## Tools

| Category | Tools |
|----------|-------|
| File ops | `read`, `write` (refuses to overwrite — use `edit` or pass `overwrite: true`), `edit` (string match or line-range), `multi_edit` |
| Search | `grep`, `glob`, `ls` |
| System | `bash` |
| Dev | `get_diagnostics` (pyright → mypy → pyflakes / tsc / node --check / shellcheck / go vet / cargo check) |
| Tasks | `task_create`, `task_update`, `task_get`, `task_list` (per-agent TODO list persisted to `agents/<name>/tasks.json`) |
| Web | `web_search`, `web_fetch` |
| Browser | `list_devices`, `capture_frame(source="webcam"\|"screen")` |
| MCP | Anything the active agent's MCP servers expose, namespaced as `mcp__<server>__<tool>` |

Read-only tools auto-execute. Dangerous tools (`write`, `edit`, `bash`, MCP tools) require user confirmation. The `bash` whitelist auto-approves safe commands (`ls`, `pwd`, `echo`, `git status`, `git log`, `git diff`, etc. — `cat`/`head`/`tail` intentionally excluded, the model should use `read`) plus any runner (`node`, `python`, `bash`) targeting `.skills/` paths inside the agent's own folder — so user-authored skill scripts run without confirm popups.

**Write-vs-Edit guard:** `write` refuses to overwrite an existing file and tells the model to use `edit` / `multi_edit` instead. Pass `overwrite: true` for a deliberate full replacement. Eliminates a whole category of accidental destruction.

**Path sandbox:** file tools (`read`, `write`, `edit`, `multi_edit`, `ls`, `grep`, `glob`) and the server's file endpoints resolve every path through `src/path_security.py`, which rejects traversal via `..`, absolute paths, or prefix-sibling tricks. The agent can only touch files under its own `agents/<name>/` directory.

**Reserved ports:** the `bash` tool blocks commands that bind to `8080` (llama-server — hijacking this kills your own LLM mid-turn) or `3040` (AINow itself). Dev servers should use `3000`, `5000`, `8000`, or `8888`.

### Web search

`web_search` is multi-backend with auto-detection by env var, in priority order:

1. **Tavily** (`TAVILY_API_KEY`) — free tier at [tavily.com](https://tavily.com), no credit card
2. **Serper** (`SERPER_API_KEY`) — free tier at [serper.dev](https://serper.dev), no credit card
3. **DuckDuckGo HTML scrape** — free fallback, often blocked from cloud IPs

Tavily is the recommended default. If you need scraping/crawling instead of plain search, install the Tavily, Exa, or Firecrawl MCP server from the suggestions list — they expose richer tools than the built-in `web_search`.

## Agentic Loop Features

- **Tool Registry** — Pluggable `ToolDef` system with `read_only` flag for permissions; supports dynamic registration (used by MCP)
- **Output Truncation** — Tool results capped at 32KB (keeps first 50% + last 25%)
- **Max Iterations** — 15 tool calls per turn to prevent infinite loops
- **Error Self-Healing** — Tool errors returned to LLM as strings, model adapts and retries
- **API Retry** — Exponential backoff on rate limits and 5xx errors
- **Context Compaction** — Two-layer strategy: snip old tool results first, then LLM summarization
- **Dynamic System Prompt** — Auto-injects date, platform, agent cwd, git branch, structured tool list (built-in + MCP grouped by server with descriptions)
- **Per-Agent CLAUDE.md** — Loaded from the active agent folder, re-read every turn so edits take effect live
- **Per-Agent Runtime Prefs** — `lang`, `voice`, `model`, `vision_enabled`, `ctx` all persisted in `agents/<name>/meta.json` under `preferences`. Switching agents restores the whole combo in a single llama-server reload. On server startup with no `-m` flag, the last-used model + context + vision are restored from the active agent's prefs.
- **Server-side Sessions** — Conversations auto-saved as JSON, scoped per agent
- **Hallucination guards** — Streaming detector that catches small models leaking chat-template tokens (`<|tool_call|>`, `<|tool_response|>`) as content text and aborts the turn before fake results reach the user; also strips stray `<|channel|>`, `<|im_start|>` etc. silently
- **Inline-command nudge** — If the model writes a shell command as text instead of calling the `bash` tool, the system detects the pattern, injects a correction message, and the model retries with a real tool call. Fires at most once per turn.
- **Repeated-failure breaker** — After 2 identical tool errors (keyed by tool name + error message), the system injects a "stop retrying, ask the user" directive. Prevents models from looping on the same broken command (e.g. missing API key).
- **Repetition detector** — If the assistant emits the same text window 4+ times, the stream is aborted gracefully and the turn unwinds cleanly (no stuck turn after interruption).
- **Edit tool line-range mode** — The `edit` tool supports replacing by line numbers (`line_start` + `line_end`) in addition to exact string matching. More reliable for models that paraphrase instead of copying exact text. Better error messages show candidate lines when `old_string` doesn't match.
- **GPU pre-warming** — Whisper (STT) and Kokoro (TTS) models load at FastAPI startup on CUDA, so the first WebSocket connection doesn't pay the cold-start
- **Skip-reload optimization** — `model_manager.start()` short-circuits when the requested model + vision + ctx + thinking flags match the currently-running state and llama-server is healthy. Reconnecting, toggling something back to its current value, or re-applying the same agent prefs no longer triggers a 3–7s restart.
- **Vision fallback for non-mmproj models** — when the active local model has no mmproj (or online model has no known vision family), the streaming path strips `image_url` blocks from the conversation history and replaces them with `[image omitted: current model has no vision adapter]`. Prevents cascading 500s that would otherwise brick a session after switching to a text-only model.
- **Skill-knowledge packs** — `src/skill_knowledge/*.md` files with YAML-ish frontmatter (`triggers:` substring/regex list + `tools:` tool-name list) are conditionally injected into the system prompt with priority-ranked selection: (1) error-recovery packs tied to the last-failed tool, (2) recency packs for recent tool calls, (3) intent packs matching the user message. A global token budget (default 500) prevents any one pack from crowding the prompt. Ships coding-oriented packs for pytest, file edits, git safety, debugging, binary search, sorting choice, DP, two-pointers, DFS/BFS, hash-vs-tree. Inspired by [little-coder](https://github.com/itayinbarr/little-coder).
- **Tolerant tool-call argument parser** — before dispatch, the `function.arguments` JSON goes through progressive repairs: escape literal newlines inside strings, strip trailing commas, normalize single quotes to double, quote unquoted keys, balance missing braces/brackets, and as a last resort regex-extract the first bare `{…}` object. Dramatically improves tool-call reliability on 9B-class models.
- **Task tools** (`task_create`, `task_update`, `task_get`, `task_list`) — the model can maintain a per-agent TODO list across multi-step work. Tasks persist to `agents/<name>/tasks.json`.
- **GetDiagnostics tool** — `get_diagnostics(path)` runs the right linter / type-checker for the file's language (pyright → mypy → pyflakes for `.py`, tsc for `.ts`, node --check for `.js`, shellcheck for `.sh`, go vet for `.go`, cargo check for `.rs`). Empty output = no issues.
- **Thinking-budget enforcement** — reasoning models are capped at `AINOW_THINKING_BUDGET_SEC` seconds (default 120) of `<think>` content before a graceful stop is forced. Prevents small reasoning models from hanging indefinitely inside the reasoning block.

## UI Features

- **Conversation History** — Sidebar with server-side sessions, auto-save after each turn, scoped to the active agent
- **Agent Picker** — Sidebar dropdown with `+` (new) / `×` (delete) buttons; switching swaps CLAUDE.md, sessions, MCP servers, scheduled tasks, cwd, and avatar live
- **Agent Profile Image** — Drop a `profile.jpg`/`png`/`webp` in the agent folder, it appears under the picker
- **CLAUDE.md Editor Modal** — Large code-editor-style modal with `Cmd/Ctrl+S` to save and live re-read on next turn
- **MCP Servers Modal** — Full CRUD over the active agent's MCP servers with 24 categorized one-click suggestions
- **Scheduled Tasks Modal** — Full CRUD with cron preset dropdown, live next-3-runs preview, mode radio (new session / inject), per-row Run/Toggle/Edit/Delete
- **Toast Notifications** — Out-of-band events (scheduled task fired, etc.) surface as bottom-right toasts
- **Model Picker** — Switch models from dropdown without restart (local + online). Switching to `online` kills the local llama-server to free GPU VRAM.
- **Vision Toggle** — Checkbox next to the model picker. When unchecked, reloads llama-server without `--mmproj`, freeing ~1.2 GB VRAM per model. Setting is persisted per-agent. Only shown for models that have an `mmproj` file.
- **Context Size Dropdown** — `4k / 16k / 32k / 48k / 64k / 96k / 128k / 256k / 512k / 1M` presets next to the model picker. Changing it reloads llama-server with the new `-c` value. Persisted per-agent.
- **Send Button Gating** — the chat send button and Enter key are disabled until the WebSocket is connected **and** a model is loaded. The input placeholder explains the reason (connect / model loading / no model).
- **Context Usage Badge** — `CTX <used>/<max>` pill in the toolbar with warn/danger thresholds at 60% / 85%. Updates live after each turn and resets on session clear.
- **Voice Picker** — Browse Kokoro voices (server TTS) or browser voices (speechSynthesis)
- **Stop Button** — Interrupt generation and TTS playback
- **Mic Mute** — Manual mic toggle, independent of auto-mute during TTS (distinct mic-off icon)
- **TTS Mute** — Silence speech output while keeping text streaming
- **Webcam / Screen Share** — Share webcam or screen with the agent (vision)
- **Image Attachments** — Upload, paste, or drag-and-drop images
- **Pipeline Metrics** — Real-time STT → LLM → TTS latency, token count, tokens/s per response; latency indicator pinned to the left so the CTX badge stays stable
- **Tool Confirm Dialog** — `Approve` / `Always` / `Deny`. `Always` session-whitelists the tool name so the agent doesn't ask again for that tool until reconnect.
- **Markdown Rendering** — Tables, headings, lists, code blocks (with `__` preserved in inline code spans), bold/italic
- **Adaptive Echo Cancellation** — Tracks ambient echo level for smarter barge-in detection
- **Mobile Sidebar** — Burger menu with tap-outside backdrop to close; auto-closes when picking a conversation
