# AINow

A voice agent framework with vision and browser tools. Runs entirely local — no cloud APIs required.

```bash
python main.py
# Open http://localhost:3040 in your browser
```

## How it works

Browser-based voice assistant with streaming LLM, TTS, and tool calling:

- **Browser STT** — Web Speech API for speech-to-text and turn detection (no server-side STT needed)
- **Local LLM** — Any OpenAI-compatible server (llama.cpp, Ollama, vLLM) with vision + tool calling
- **Browser TTS** — Native `speechSynthesis` API with voice picker popup, preview, and localStorage persistence. Prioritizes Online (neural) voices. Falls back to local Kokoro when `SERVER_TTS` is set
- **Agent** — LLM → TTS → Browser audio pipeline, owns conversation history
- **Browser Tools** — LLM can capture webcam/screen frames and see them via vision
- **Tool Confirmation** — Dangerous tools (write, edit, bash) require user approval via in-UI confirm/deny buttons
- **Text + Voice Input** — Type messages or speak; attach images via upload, paste, or drag-and-drop
- **Multi-Language** — Language selector (EN/FR/ES/IT/PT/ZH/HI) switches STT, TTS voice, and server language in real time
- **Barge-In** — Client-side Silero VAD (voice activity detection) interrupts bot speech instantly when the user starts talking. Mic stays live during TTS playback (audio send is paused, not the mic track) so VAD can detect speech; browser echo cancellation prevents false triggers
- **`process_event(state, event) → (state, actions)`** — the entire state machine in ~30 lines

Everything streams. LLM tokens feed TTS immediately, TTS audio plays in the browser. Barge-in cancels everything instantly.

```
LISTENING ──EndOfTurn──→ RESPONDING ──Done──→ LISTENING
    ↑                        │
    └────StartOfTurn─────────┘  (barge-in)
```

## UI Features

- **Voice Picker** — Glass-style popup (🎙️ button) lists available browser voices, filtered by language with Online voices first. Click to select, play button to preview. Choice persists across sessions via localStorage
- **TTS Mute** — 🔊 toggle to mute/unmute speech output
- **Webcam / Screen Share** — Toggle buttons to share webcam or screen with the agent
- **Custom System Prompt** — Expandable textarea (fills available vertical space) to customize the agent's behavior
- **Image Attachments** — Upload, paste, or drag-and-drop images into the chat
- **Pipeline Visualization** — Real-time STT → LLM → TTS → Playing stage indicators with latency metrics

## Tools

The agent has access to:

| Category | Tools |
|----------|-------|
| File ops | `read`, `write`, `edit`, `multi_edit` |
| Search | `grep`, `glob`, `ls` |
| System | `bash` |
| Browser | `list_devices`, `capture_frame(source="webcam"\|"screen")` |

Browser tools execute client-side via WebSocket — the LLM can see through the webcam or capture the screen. Dangerous tools require user confirmation before execution.

## CLI Usage

```bash
python main.py                         # default: 9B model, server-only mode
python main.py --model 4b              # start with 4B model
python main.py --model 27b             # start with 27B model
python main.py +1234567890             # outbound call to phone number
python main.py --model 9b +1234567890  # specific model + outbound call
```

### Model sizes

| Alias | Model | Quant | Size |
|-------|-------|-------|------|
| `0.8b` | Qwen 3.5 0.8B | Q8_0 | ~1 GB |
| `2b` | Qwen 3.5 2B | Q4_K_M | ~1.5 GB |
| `4b` | Qwen 3.5 4B | Q4_K_M | ~2.8 GB |
| `9b` | Qwen 3.5 9B | UD-Q4_K_XL | ~6 GB |
| `27b` | Qwen 3.5 27B | UD-IQ3_XXS | ~11 GB |

All models include vision (mmproj). The model manager starts llama-server automatically.

### Start scripts

Quick-launch scripts for each model size:

```bash
start-0.8b.cmd    # Qwen 0.8B
start-2b.cmd      # Qwen 2B
start-4b.cmd      # Qwen 4B
start-9b.cmd      # Qwen 9B (default)
start-27b.cmd     # Qwen 27B
```

Pass additional arguments after the script name, e.g. `start-9b.cmd +1234567890` for outbound calls.

### Optimal 16 GB VRAM

```bash
optimal16GB.bat
```

Best quality combination for a 16 GB GPU: **Qwen 9B UD-Q4_K_XL** (~6 GB) + **Whisper large-v3** (~3 GB) + vision mmproj (~1 GB). Leaves ~6 GB headroom for KV cache and context. Fast LLM inference with the best available STT accuracy.

### Comfort 16 GB VRAM

```bash
comfort16GB.bat
```

Maximum context for long conversations: **Qwen 9B UD-Q4_K_XL** (~6 GB) + **Whisper medium** (~1.5 GB) + 256K context with Q4_0 KV cache. Uses ~13 GB, leaving headroom. Best for extended sessions where context length matters more than STT accuracy.

### Budget 8 GB VRAM

```bash
start-4b.cmd
```

Good balance for 8 GB GPUs: **Qwen 4B Q4_K_M** (~2.8 GB) + **Whisper small** (~0.5 GB) + vision mmproj (~0.6 GB). Leaves room for KV cache. Smart enough for conversation and tool use, fast inference.

### Fastest (low VRAM)

```bash
start-0.8b.cmd
```

Fastest possible response times with **Qwen 0.8B Q8_0** (~1 GB) + **Whisper small** (~0.5 GB). Minimal VRAM usage, near-instant inference. Not very smart — best for testing latency or low-end hardware.

## Project structure

```
ainow/
  types.py              # Immutable state, events, actions
  state.py              # Pure state machine (~30 lines)
  conversation.py       # Main event loop (browser + Twilio)
  agent.py              # LLM → TTS → Player pipeline
  log.py                # Colored logging
  server.py             # FastAPI endpoints
  tracer.py             # Per-call JSON trace logging (/tmp/ainow/)
  static/index.html     # Browser UI (single-file SPA)
  services/
    llm.py              # OpenAI-compatible streaming + tool calling + vision
    tools.py            # Tool definitions + execution (file ops, search, bash, browser)
    model_manager.py    # Auto-download + launch llama-server
    local_tts.py        # Kokoro TTS (local, multi-language)
    fish_tts.py         # Fish Speech TTS (self-hosted, optional)
    tts.py              # ElevenLabs WebSocket streaming (optional)
    tts_pool.py         # TTS connection pool
    browser_player.py   # Audio playback to browser via WebSocket
    whisper_stt.py      # Whisper STT (server-side, optional)
    player.py           # Audio playback to Twilio (optional)
    flux.py             # Deepgram Flux STT (optional)
    twilio_client.py    # Twilio calls (optional)
```

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
cp .env.example .env   # edit as needed
python main.py
```

The model manager will download the default model and start llama-server automatically. To use your own LLM server instead, set `LLM_BASE_URL` in `.env`.

### Environment variables

```bash
# LLM (auto-managed by default; set these to use an external server)
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=not-needed
LLM_MODEL=Qwen3.5-9B-UD-Q4_K_XL.gguf

# TTS mode (default: browser speechSynthesis)
SERVER_TTS=1              # use server-side Kokoro TTS instead of browser TTS
LOCAL_TTS_VOICE=af_heart  # Kokoro voice name (when using server TTS)

# Optional: Fish Speech TTS (self-hosted)
FISH_TTS_URL=http://localhost:8082

# Optional: ElevenLabs TTS (cloud)
ELEVENLABS_API_KEY=your_key

# Optional: Whisper STT (server-side, faster-whisper on CUDA)
WHISPER_MODEL=large-v3        # Whisper model size: tiny, base, small, medium, large-v3

# Optional: Deepgram STT (if not set, uses browser Web Speech API)
DEEPGRAM_API_KEY=your_key

# Model directory (default: ./models)
MODELS_DIR=/path/to/models
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Browser UI |
| `WebSocket /ws` | Browser voice session |
| `GET /trace/latest` | Latest call trace JSON |
| `GET /bench/ttft` | TTFT benchmark across models |
| `GET/POST /twiml` | Twilio TwiML webhook |
| `GET /call/{phone}` | Initiate outbound call |

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
