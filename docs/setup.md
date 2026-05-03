# Setup, environment & security

## Setup

Requires Python 3.9+ and a CUDA GPU (for local models).

Models and llama-server are **auto-downloaded on first run**. GGUF model files are fetched from HuggingFace, and the llama-server CUDA binary is downloaded from the latest llama.cpp GitHub release.

```bash
pip install -r requirements.txt
cp .env.example .env   # edit as needed
python main.py
```

## Custom models

Add any GGUF model by setting an env var pointing to a directory:

```bash
MODEL_MYMODEL=/path/to/gguf-directory  # Auto-detects .gguf and mmproj files
# Or explicit:
MODEL_MYMODEL=/path/model.gguf;/path/mmproj.gguf;Display Name
```

Then: `python main.py -m mymodel`

### Per-model overrides

For models that don't fit your VRAM at the default context size, you can override the llama-server context length and GPU layer offload per model:

```bash
MODEL_HERETIC=C:\path\heretic.gguf;C:\path\mmproj.gguf;Gemma 4 26B Heretic
MODEL_HERETIC_CTX=131072    # 128K context (overrides global -c)
MODEL_HERETIC_NGL=24        # offload 24 layers to GPU (-ngl)
```

Then: `python main.py -m heretic` (or pick from the UI dropdown).

## Recommended setups

**Best experience** is with **server-side STT/TTS** (Whisper + Kokoro on GPU) — better transcript quality, lower latency, and more reliable barge-in interruptions than browser-based alternatives.

### With a GPU (best experience)

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

### Without a GPU

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

## Environment variables

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

# Hardening (see Security section)
# AINOW_ADMIN_TOKEN=<long random>  # unlocks runtime endpoints from non-localhost
# AINOW_ENABLE_DEBUG_ROUTES=1      # expose /trace/latest, /bench/ttft, /api/test-thinking
# AINOW_TRACE_DIR=/var/log/ainow   # default: <tempfile.gettempdir()>/ainow

# Reasoning / thinking cap for small reasoning models (default 120s)
# AINOW_THINKING_BUDGET_SEC=120

# Per-bash-call memory cap (default 8 GB). On timeout or when this cap is
# exceeded, the entire process tree is killed (parent + descendants) to
# stop runaway test/build processes from eating system RAM.
# AINOW_BASH_MEMORY_CAP_GB=8

# TurboQuant KV cache compression (default: turbo3 = 3.8x smaller KV)
# KV_CACHE_TYPE=turbo3             # turbo2/turbo3/turbo4 require a TurboQuant llama-server build
                                   # falls back to q4_0 automatically on mainline binaries
```

### Optional system tools

- **`uv` / `uvx`** — required only if you want to install Python-based MCP servers (`mcp-server-fetch`, `mcp-server-time`, `mcp-server-git`, etc.). Install from [astral.sh/uv](https://astral.sh/uv).
- **`node` / `npx`** — required only if you want to install npm-based MCP servers. Install from [nodejs.org](https://nodejs.org).

## Security / Hardening

AINow is designed to run on `localhost` by default. If you expose it beyond that, here's what the hardening layer does and how to configure it.

- **Path sandbox** (`src/path_security.py`) — every file access (agent tools, `/api/files/*`, tar.gz import) goes through `resolve_within_base(base, rel)` which rejects traversal via `..`, absolute paths, prefix-sibling tricks, and control characters. Tests in `tests/test_tools_paths.py` + `tests/test_server_security.py`.
- **Admin gate** on state-changing runtime endpoints (`/api/runtime`, `/api/runtime/settings`, `/api/eject-model`, `/api/models/{alias}`, `/api/agents/import`): allowed from `127.0.0.1`/`::1`/`localhost` unconditionally, or from remote clients that present a header `x-ainow-admin-token: $AINOW_ADMIN_TOKEN`.
- **Debug routes** (`/trace/latest`, `/bench/ttft`, `/api/test-thinking`) return 404 unless `AINOW_ENABLE_DEBUG_ROUTES=1` is set (and the admin check still applies).
- **Tar.gz import** caps: 64 MB archive, 5 MB per file, 2000 members max; symlinks and path-traversal entries rejected before extraction.
- **Session ID validation** — every `session_id` coming from REST, WebSocket, or internal callers is matched against `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` before touching the filesystem.
- **WS control-type whitelist** — unknown or malformed control messages are rejected with a warning instead of flowing through.
- **Graceful drain** — on SIGTERM the server rejects new WS connections and waits for active turns to finish.

Run the tests with `pytest tests/` (requires `pytest`; dependency-light, no heavy project imports needed).

## Project structure

```
src/
  server.py                      # FastAPI endpoints + WebSocket + lifecycle hooks
  conversation.py                # Main event loop, registers in live_conversations
  agent.py                       # LLM -> TTS pipeline (per-WebSocket)
  state.py                       # Pure state machine
  types.py                       # Immutable state, events, actions
  log.py                         # Colored logging
  tracer.py                      # Per-session JSON trace logging (AINOW_TRACE_DIR)
  path_security.py               # is_within_base / resolve_within_base sandbox helpers
  cli.py                         # `python -m src.cli` — headless CLI agent
  tui_textual.py                 # Textual-based TUI (default for `--interactive`)
  tui_fullscreen.py              # prompt_toolkit full-screen TUI (experimental)
  skill_knowledge/               # Conditionally-injected system-prompt packs
    __init__.py                  # Loader: parses frontmatter, selects by triggers/tools
    *.md                         # One guidance pack per file (testing, git, edits, …)
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
tests/                           # pytest suite (dependency-light)
  test_tools_paths.py            # resolve_within_base against traversal
  test_server_security.py        # is_within_base prefix-sibling check
  test_model_manager.py          # _should_load_mmproj vision/audio-llm logic
agents/                          # Per-agent workspaces (gitignored by default)
  default/
    CLAUDE.md
    meta.json                    # display_name, mcp_servers
    sessions/*.json
    scheduled_tasks.json
    profile.jpg                  # optional avatar
main.py                          # CLI entry point (web server)
```
