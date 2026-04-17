# AINow

A conversational AI framework with voice, vision, per-agent workspaces, MCP tool ecosystem, and scheduled cron-AI tasks. Runs entirely local — no cloud APIs required.

![AINow UI](docs/ui-screenshot.png)

```bash
python main.py
# Open http://localhost:3040
```

## How it works

Browser-based voice assistant with streaming LLM, TTS, tool calling, and a sophisticated agentic loop:

- **Server STT (Whisper)** — Local faster-whisper with Silero VAD for speech detection. Higher transcript quality and better barge-in handling than browser STT. Falls back to browser Web Speech API if disabled.
- **Server TTS (Kokoro)** — Local Kokoro TTS streamed via Web Audio API. Lower latency and better barge-in than browser speechSynthesis. Supports EN/FR/ES/IT/PT/ZH/HI.
- **Local LLM** — Any OpenAI-compatible server (llama.cpp, Ollama, vLLM) with vision + tool calling
- **Silero VAD** — On-device ONNX voice activity detection for reliable barge-in
- **Per-Agent Workspaces** — Each agent has its own folder with `CLAUDE.md`, sessions, MCP servers, scheduled tasks, profile image, and isolated tool cwd. Switch live from the UI.
- **MCP (Model Context Protocol)** — Plug in any of the 5,400+ MCP servers in the ecosystem. Browse 24 curated suggestions in the UI for one-click install.
- **Scheduled Tasks (Cron AI)** — Per-agent recurring or one-time prompts that fire on cron schedules. Run as new sessions or inject into the active chat.
- **Tool Calling** — File ops, bash, web search (Tavily/Serper/DuckDuckGo), webcam/screen capture, plus any MCP tool the active agent loads
- **Agentic Loop** — Tool registry, error self-healing, max iterations guard, output truncation, hallucination guards
- **Context Management** — Auto-snip old tool results, LLM-based summarization at 70% of context limit
- **Multi-Language** — Language selector (EN/FR/ES/IT/PT/ZH/HI) switches STT and TTS in real time
- **Markdown Rendering** — Full markdown in chat: tables, lists, code blocks (with `__` preserved in inline code), headings, bold/italic

Everything streams. LLM tokens feed TTS immediately. Barge-in interrupts the bot mid-sentence.

```
LISTENING ──EndOfTurn──→ RESPONDING ──Done──→ LISTENING
    ↑                        │
    └────StartOfTurn─────────┘  (barge-in via Silero VAD)
```

## Usage

```bash
python main.py                    # default: Qwen 9B
python main.py -m 4b              # smaller model (less VRAM)
python main.py -m 27b             # larger model (more quality)
python main.py -m gemma           # custom Gemma 4 (uncensored, via MODEL_GEMMA env)
python main.py -m heretic         # custom Gemma 4 26B (uncensored, via MODEL_HERETIC env)
python main.py -m online          # Cloud provider (OpenRouter, OpenAI, etc.)
```

You can also switch models from the UI dropdown at runtime — no restart needed.

### Available models

| Flag | Model | Size | Notes |
|------|-------|------|-------|
| `-m 0.8b` | Qwen 3.5 0.8B Q8_0 | ~1 GB | Fastest, minimal quality |
| `-m 2b` | Qwen 3.5 2B Q4_K_M | ~1.5 GB | Budget GPU |
| `-m 4b` | Qwen 3.5 4B Q4_K_M | ~2.8 GB | Good for 8 GB VRAM |
| `-m 9b` | Qwen 3.5 9B UD-Q4_K_XL | ~6 GB | **Default**, best balance |
| `-m 27b` | Qwen 3.5 27B UD-IQ3_XXS | ~11 GB | Highest quality dense |
| `-m 35b` | Qwen 3.6 35B A3B UD-Q2_K_XL | ~13 GB | MoE, 3B active — fast for its size |
| `-m 35b-iq1` | Qwen 3.6 35B A3B UD-IQ1_M | ~10 GB | MoE, smaller quant — fits 16 GB VRAM |
| `-m 35b-agg` | Qwen 3.6 35B Aggressive IQ2_M (uncensored) | ~12 GB | MoE + vision mmproj (HauhauCS) |
| `-m gemma` | Gemma 4 E4B Aggressive (uncensored) | ~5 GB | Custom — set `MODEL_GEMMA` |
| `-m heretic` | Gemma 4 26B Heretic (uncensored) | ~13 GB | Custom — set `MODEL_HERETIC` + `_CTX` + `_NGL` |
| `-m online` | Any OpenAI-compatible API | 0 GB | Cloud, needs API key |

All local Qwen models include vision (mmproj). The model manager starts llama-server automatically and downloads models from HuggingFace on first run.

### Custom models

Add any GGUF model by setting an env var pointing to a directory:

```bash
MODEL_MYMODEL=/path/to/gguf-directory  # Auto-detects .gguf and mmproj files
# Or explicit:
MODEL_MYMODEL=/path/model.gguf;/path/mmproj.gguf;Display Name
```

Then: `python main.py -m mymodel`

#### Per-model overrides

For models that don't fit your VRAM at the default context size, you can override
the llama-server context length and GPU layer offload per model:

```bash
MODEL_HERETIC=C:\path\heretic.gguf;C:\path\mmproj.gguf;Gemma 4 26B Heretic
MODEL_HERETIC_CTX=131072    # 128K context (overrides global -c)
MODEL_HERETIC_NGL=24        # offload 24 layers to GPU (-ngl)
```

Then: `python main.py -m heretic` (or pick from the UI dropdown).

### Recommended setups

**Best experience** is with **server-side STT/TTS** (Whisper + Kokoro on GPU) — better transcript quality, lower latency, and more reliable barge-in interruptions than browser-based alternatives.

#### With a GPU (best experience)

| VRAM | LLM | STT/TTS | Notes |
|------|-----|---------|-------|
| **8 GB** | Gemma 4 E4B | Whisper small + Kokoro | Best budget setup. Fast, uncensored. Set `MODEL_GEMMA`. |
| **16 GB** | Gemma 4 26B Q3 | Whisper medium + Kokoro | Best overall. 26B uncensored with vision. Set `MODEL_HERETIC` + `_CTX=32768` + `_NGL=24`. Disable vision to free ~1.2 GB if needed. |
| **24 GB** | Qwen 27B or Gemma 26B | Whisper medium + Kokoro | Full quality, 64K+ context, vision enabled. |

For GPU setups, enable server STT/TTS in `.env`:
```bash
# BROWSER_STT=1           # keep commented for server Whisper
WHISPER_MODEL=medium       # or small (less VRAM) / large-v3 (best quality)
SERVER_TTS=1
LOCAL_TTS_VOICE=1
```

#### Without a GPU

| Setup | LLM | STT/TTS |
|-------|-----|---------|
| **Free** | `google/gemma-4-31b-it:free` via OpenRouter (`-m online2`) | Browser (Chrome Web Speech API + speechSynthesis) |
| **Paid** | `google/gemini-3.1-flash-lite-preview` via OpenRouter (`-m online`) | Browser |

For no-GPU setups, use browser STT/TTS in `.env`:
```bash
BROWSER_STT=1
# SERVER_TTS=1            # keep commented
# LOCAL_TTS_VOICE=1       # keep commented
```

Browser STT/TTS works fine but has limitations: Chrome's Web Speech API occasionally drops recognition, browser voices sound more robotic, and barge-in timing is less precise than with Whisper + Kokoro.

### VRAM guidelines (quick reference)

| VRAM | Recommended | Command |
|------|-------------|---------|
| 8 GB | Gemma 4 E4B | `python main.py -m gemma` |
| 16 GB | Gemma 4 26B Heretic | `python main.py -m heretic` |
| 24 GB | Qwen 27B or Qwen 3.6 35B A3B | `python main.py -m 27b` / `-m 35b` |
| No GPU | Gemma 31B free (OpenRouter) | `python main.py -m online2` |

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
| File ops | `read`, `write`, `edit` (string match or line-range), `multi_edit` |
| Search | `grep`, `glob`, `ls` |
| System | `bash` |
| Web | `web_search`, `web_fetch` |
| Browser | `list_devices`, `capture_frame(source="webcam"\|"screen")` |
| MCP | Anything the active agent's MCP servers expose, namespaced as `mcp__<server>__<tool>` |

Read-only tools auto-execute. Dangerous tools (`write`, `edit`, `bash`, MCP tools) require user confirmation. The `bash` whitelist auto-approves safe commands (`ls`, `git status`, `pwd`, `cat`, etc.) plus any runner (`node`, `python`, `bash`) targeting `.skills/` paths inside the agent's own folder — so user-authored skill scripts run without confirm popups.

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
- **Edit tool line-range mode** — The `edit` tool supports replacing by line numbers (`line_start` + `line_end`) in addition to exact string matching. More reliable for models that paraphrase instead of copying exact text. Better error messages show candidate lines when `old_string` doesn't match.
- **GPU pre-warming** — Whisper (STT) and Kokoro (TTS) models load at FastAPI startup on CUDA, so the first WebSocket connection doesn't pay the cold-start

## UI Features

- **Conversation History** — Sidebar with server-side sessions, auto-save after each turn, scoped to the active agent
- **Agent Picker** — Sidebar dropdown with `+` (new) / `×` (delete) buttons; switching swaps CLAUDE.md, sessions, MCP servers, scheduled tasks, cwd, and avatar live
- **Agent Profile Image** — Drop a `profile.jpg`/`png`/`webp` in the agent folder, it appears under the picker
- **CLAUDE.md Editor Modal** — Large code-editor-style modal with `Cmd/Ctrl+S` to save and live re-read on next turn
- **MCP Servers Modal** — Full CRUD over the active agent's MCP servers with 24 categorized one-click suggestions
- **Scheduled Tasks Modal** — Full CRUD with cron preset dropdown, live next-3-runs preview, mode radio (new session / inject), per-row Run/Toggle/Edit/Delete
- **Toast Notifications** — Out-of-band events (scheduled task fired, etc.) surface as bottom-right toasts
- **Model Picker** — Switch models from dropdown without restart (local + online). Switching to `online` kills the local llama-server to free GPU VRAM.
- **Vision Toggle** — Checkbox next to the model picker. When unchecked, reloads llama-server without `--mmproj`, freeing ~1.2 GB VRAM per model. Setting is persisted per-agent.
- **Context Size Dropdown** — `4k / 16k / 32k / 48k / 64k / 96k / 128k / 256k` presets next to the model picker. Changing it reloads llama-server with the new `-c` value. Persisted per-agent.
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

## Setup

Requires Python 3.9+ and a CUDA GPU (for local models).

Models and llama-server are **auto-downloaded on first run**. GGUF model files are fetched from HuggingFace, and the llama-server CUDA binary is downloaded from the latest llama.cpp GitHub release.

```bash
pip install -r requirements.txt
cp .env.example .env   # edit as needed
python main.py
```

### Environment variables

```bash
# Server
PORT=3040                         # Web UI port

# llama-server
LLAMA_SERVER_EXE=llama-server     # path to binary (auto-downloaded if missing)
LLAMA_SERVER_PORT=8080
MODELS_DIR=~/models               # base dir for GGUF files

# LLM
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=not-needed

# Online mode (-m online): any OpenAI-compatible API
ONLINE_BASE_URL=https://openrouter.ai/api/v1
ONLINE_API_KEY=your_key
ONLINE_MODEL=google/gemini-3.1-flash-lite-preview

# STT (default: browser Web Speech API)
BROWSER_STT=1
# Server-side Whisper STT (only used when BROWSER_STT is unset):
# WHISPER_MODEL=small             # base / small / medium / large-v3
# WHISPER_DEVICE=cpu              # force CPU to free ~500 MB VRAM (default: auto cuda)

# TTS (default: server Kokoro)
SERVER_TTS=1
LOCAL_TTS_VOICE=1                 # Kokoro TTS

# Audio LLM mode (experimental — for models that accept audio input directly)
# AUDIO_LLM=1                     # send audio_url to LLM, skip STT

# Web search (priority order; all are optional)
# TAVILY_API_KEY=tvly-...         # free tier at tavily.com
# SERPER_API_KEY=...              # free tier at serper.dev

# Custom models
# MODEL_MYMODEL=/path/to/gguf-directory
# MODEL_MYMODEL_CTX=131072        # per-model context override
# MODEL_MYMODEL_NGL=24            # per-model GPU layer offload
```

### Optional system tools

- **`uv` / `uvx`** — required only if you want to install Python-based MCP servers (`mcp-server-fetch`, `mcp-server-time`, `mcp-server-git`, etc.). Install from [astral.sh/uv](https://astral.sh/uv).
- **`node` / `npx`** — required only if you want to install npm-based MCP servers. Install from [nodejs.org](https://nodejs.org).

## Project structure

```
src/
  server.py                      # FastAPI endpoints + WebSocket + lifecycle hooks
  conversation.py                # Main event loop, registers in live_conversations
  agent.py                       # LLM -> TTS pipeline (per-WebSocket)
  state.py                       # Pure state machine
  types.py                       # Immutable state, events, actions
  log.py                         # Colored logging
  tracer.py                      # Per-session JSON trace logging
  static/
    index.html                   # Browser UI (single-file SPA)
    lib/                         # Silero VAD + ONNX runtime (offline)
  services/
    llm.py                       # OpenAI streaming + dynamic tool schemas + per-agent sessions
                                 #   + hallucination guards (template token detector)
    tools.py                     # Built-in tool registry + execution + multi-backend web_search
    model_manager.py             # Auto-download + launch llama-server with per-model overrides
    local_tts.py                 # Kokoro TTS (server-side, streamed)
    browser_player.py            # Audio playback via WebSocket
    whisper_stt.py               # Whisper STT (server-side, optional)
    agents.py                    # Per-agent storage: CLAUDE.md, sessions, mcp_servers,
                                 #   scheduled_tasks, profile image, active-agent state
    mcp.py                       # MCP integration: long-lived stdio server tasks,
                                 #   per-agent registration into TOOL_REGISTRY
    live_conversations.py        # Registry of connected WebSocket sessions
                                 #   (used by scheduler to inject prompts / push notifications)
    scheduler.py                 # Cron AI: per-agent scheduled tasks, headless execution,
                                 #   inject-into-active mode, run-now
agents/                          # Per-agent workspaces (gitignored by default)
  default/
    CLAUDE.md
    meta.json                    # display_name, mcp_servers
    sessions/*.json
    scheduled_tasks.json
    profile.jpg                  # optional avatar
main.py                          # CLI entry point
```

## License

GNU Affero General Public License v3.0 (AGPL-3.0).

AINow is free software: you can use, modify, and redistribute it under the terms of the AGPL-3.0. A key implication: if you run a modified version of AINow over a network (including as a SaaS or internal hosted service), you must make the complete source code of your modified version available to its users. See [LICENSE](LICENSE) for the full terms.

For a commercial license (e.g. to build closed-source products or hosted services without the AGPL source-sharing obligation), contact the author.
