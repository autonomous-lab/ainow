# AINow

A conversational AI framework with voice, vision, and tool calling. Runs entirely local -- no cloud APIs required.

```bash
python main.py
# Open http://localhost:3040
```

## How it works

Browser-based voice assistant with streaming LLM, TTS, and tool calling:

- **Browser STT** -- Web Speech API for speech-to-text and turn detection
- **Local LLM** -- Any OpenAI-compatible server (llama.cpp, Ollama, vLLM) with vision + tool calling
- **Browser TTS** -- Native `speechSynthesis` with voice picker, preview, and language support. Falls back to local Kokoro when `SERVER_TTS` is set
- **Tool Calling** -- Web search, web fetch, file ops, bash, webcam/screen capture
- **Conversation History** -- Sidebar with saved conversations (localStorage)
- **Text + Voice Input** -- Type messages or speak; attach images via upload, paste, or drag-and-drop
- **Multi-Language** -- Language selector (EN/FR/ES/IT/PT/ZH/HI) switches STT and TTS in real time
- **Barge-In** -- VAD interrupts bot speech when user starts talking

Everything streams. LLM tokens feed TTS immediately. Barge-in cancels everything instantly.

```
LISTENING ──EndOfTurn──→ RESPONDING ──Done──→ LISTENING
    ↑                        │
    └────StartOfTurn─────────┘  (barge-in)
```

## Usage

```bash
python main.py                    # default: Qwen 9B, browser STT/TTS
python main.py -m 4b              # smaller model (less VRAM)
python main.py -m 27b             # larger model (more quality)
python main.py -m gemma           # Gemma 4B (uncensored)
python main.py -m online          # Gemini Flash Lite via OpenRouter (no local GPU needed)
```

### Available models

| Flag | Model | Size | Notes |
|------|-------|------|-------|
| `-m 0.8b` | Qwen 3.5 0.8B Q8_0 | ~1 GB | Fastest, minimal quality |
| `-m 2b` | Qwen 3.5 2B Q4_K_M | ~1.5 GB | Budget GPU |
| `-m 4b` | Qwen 3.5 4B Q4_K_M | ~2.8 GB | Good for 8 GB VRAM |
| `-m 9b` | Qwen 3.5 9B UD-Q4_K_XL | ~6 GB | **Default**, best balance |
| `-m 27b` | Qwen 3.5 27B UD-IQ3_XXS | ~11 GB | Highest quality |
| `-m gemma` | Gemma 4B Uncensored Q4_K_M | ~5 GB | No content filters |
| `-m online` | Gemini Flash Lite (OpenRouter) | 0 GB | Cloud, needs API key |

All local models include vision (mmproj). The model manager starts llama-server automatically.

### VRAM guidelines

| VRAM | Recommended | Command |
|------|-------------|---------|
| 4 GB | Qwen 0.8B | `python main.py -m 0.8b` |
| 8 GB | Qwen 4B | `python main.py -m 4b` |
| 16 GB | Qwen 9B (default) | `python main.py` |
| 24 GB | Qwen 27B | `python main.py -m 27b` |
| No GPU | Gemini online | `python main.py -m online` |

## Tools

| Category | Tools |
|----------|-------|
| File ops | `read`, `write`, `edit`, `multi_edit` |
| Search | `grep`, `glob`, `ls` |
| System | `bash` |
| Web | `web_search`, `web_fetch` |
| Browser | `list_devices`, `capture_frame(source="webcam"|"screen")` |

Dangerous tools (write, edit, bash) require user confirmation before execution.

## UI Features

- **Conversation History** -- Sidebar with saved conversations, auto-save, create/delete
- **Voice Picker** -- Browse and preview available browser voices, filtered by language
- **Stop Button** -- Instantly stop generation and TTS playback
- **Mic Mute** -- Manual mic toggle, independent of auto-mute during TTS
- **TTS Mute** -- Silence speech output while keeping text streaming
- **Webcam / Screen Share** -- Share webcam or screen with the agent
- **Custom System Prompt** -- Editable system prompt, applied live
- **Image Attachments** -- Upload, paste, or drag-and-drop images
- **Pipeline Metrics** -- Real-time STT -> LLM -> TTS latency, token count, tokens/s per response
- **Markdown Rendering** -- Headings, lists, code blocks, bold/italic in responses
- **Clear Session** -- Reset conversation context without disconnecting

## Setup

Requires Python 3.9+ and a CUDA GPU (for local models).

Models and llama-server are **auto-downloaded on first run**. The selected GGUF model files are fetched from HuggingFace, and the llama-server CUDA binary is downloaded from the latest llama.cpp GitHub release. No manual setup needed beyond installing dependencies.

```bash
pip install -r requirements.txt
cp .env.example .env   # edit as needed
python main.py
```

### Environment variables

```bash
# Server
PORT=3040                         # Web UI port (default: 3040)

# LLM (auto-managed by default; set to use an external server)
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=not-needed
LLM_MODEL=Qwen3.5-9B-UD-Q4_K_XL.gguf

# Online mode (no local GPU)
OPENROUTER_API_KEY=your_key       # for -m online

# STT mode (default: browser Web Speech API)
BROWSER_STT=1                     # explicit browser STT
WHISPER_MODEL=large-v3            # server-side Whisper STT

# TTS mode (default: browser speechSynthesis)
SERVER_TTS=1                      # use server-side TTS instead
LOCAL_TTS_VOICE=1                 # local Kokoro TTS (no API key)
FISH_TTS_URL=http://localhost:8082  # Fish Speech TTS
```

## Project structure

```
ainow/
  server.py             # FastAPI endpoints
  conversation.py       # Main event loop (browser voice)
  agent.py              # LLM -> TTS -> Player pipeline
  state.py              # Pure state machine (~30 lines)
  types.py              # Immutable state, events, actions
  log.py                # Colored logging
  tracer.py             # Per-session JSON trace logging
  static/index.html     # Browser UI (single-file SPA)
  services/
    llm.py              # OpenAI streaming + tool calling + vision
    tools.py            # Tool definitions + execution
    model_manager.py    # Launch llama-server with model configs
    local_tts.py        # Kokoro TTS (local)
    fish_tts.py         # Fish Speech TTS (self-hosted)
    browser_player.py   # Audio playback via WebSocket
    whisper_stt.py      # Whisper STT (server-side)
main.py                 # CLI entry point
```

## License

Apache 2.0
