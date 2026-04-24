# AINow

**Local-first AI agent framework** — chat, voice, vision, coding, scheduled background tasks, MCP. Runs entirely on your GPU, no cloud APIs required.

![AINow UI](docs/ui-screenshot.png)

```bash
python main.py
# Open http://localhost:3040
```

## Documentation

| Topic | Docs |
|---|---|
| **Headless CLI / TUI** — slash commands, keyboard shortcuts, `@path` expansion | [docs/cli.md](docs/cli.md) |
| **Agents, MCP, scheduled tasks, tools, agentic loop, UI features** | [docs/agents-mcp.md](docs/agents-mcp.md) |
| **Setup, env vars, custom models, security, project structure** | [docs/setup.md](docs/setup.md) |

## Modes of use

AINow is a general-purpose local agent that adapts to several workflows out of the box:

- **Text chat** — a fast, local ChatGPT-style assistant with markdown rendering, streaming, per-conversation memory, and file/image/audio attachments.
- **Local coding agent** — file tools (`read`, `write`, `edit`, `multi_edit`, `grep`, `glob`, `ls`), `bash` with a safe-command whitelist, git-aware workspace context, and skill-knowledge packs that inject focused guidance for testing / editing / debugging / git workflows.
- **Voice assistant** — STT (Whisper or browser), TTS (Kokoro or browser), Silero VAD for barge-in.
- **Vision** — webcam / screen capture, image paste/upload, `mmproj`-based local vision on Qwen/Gemma, or cloud vision via Gemini/GPT-4/Claude through an OpenAI-compatible endpoint.
- **Scheduled background tasks (Cron AI)** — per-agent recurring prompts (daily news digest, periodic code checks, etc.) running headless and saving results as chats.
- **Per-agent workspaces** — each agent has its own folder with `CLAUDE.md`, sessions, MCP servers, scheduled tasks, profile image, and isolated tool cwd.
- **MCP ecosystem** — plug any of the 5,400+ MCP servers in (24 curated one-click suggestions in the UI): GitHub, Tavily, Linear, Postgres, Playwright, etc.

## How it works

Local-first agent runtime with a browser UI **and** a Textual CLI, streaming LLM, tool calling, and a sophisticated agentic loop:

- **Server STT (Whisper)** — Local faster-whisper with Silero VAD. Falls back to browser Web Speech API.
- **Server TTS (Kokoro)** — Local Kokoro TTS streamed via Web Audio API. Supports EN/FR/ES/IT/PT/ZH/HI.
- **Local LLM** — Any OpenAI-compatible server (llama.cpp, Ollama, vLLM) with vision + tool calling.
- **Silero VAD** — On-device ONNX voice activity detection for reliable barge-in.
- **MCP** — Plug in any of the 5,400+ MCP servers.
- **Scheduled Tasks (Cron AI)** — Per-agent recurring or one-time prompts.
- **Tool Calling** — File ops, bash, web search (Tavily/Serper/DuckDuckGo), webcam/screen capture, plus any MCP tool the active agent loads.
- **Agentic Loop** — Tool registry, error self-healing, max iterations guard, output truncation, hallucination guards.
- **Context Management** — Auto-snip old tool results, LLM-based summarization at 70% of context limit.

Everything streams. LLM tokens feed TTS immediately. Barge-in interrupts the bot mid-sentence.

```
LISTENING ──EndOfTurn──→ RESPONDING ──Done──→ LISTENING
    ↑                        │
    └────StartOfTurn─────────┘  (barge-in via Silero VAD)
```

## Usage

```bash
# Web UI
python main.py                    # default: Qwen 9B
python main.py -m 4b              # smaller model (less VRAM)
python main.py -m 27b             # larger Qwen 3.5 dense
python main.py -m 35b-agg         # Qwen 3.6 35B MoE (uncensored, with vision)
python main.py -m 27b-iq2         # Qwen 3.6 27B (local LM Studio path)
python main.py -m gemma           # custom Gemma 4 (uncensored, via MODEL_GEMMA env)
python main.py -m heretic         # custom Gemma 4 26B (uncensored, via MODEL_HERETIC env)
python main.py -m online          # Cloud provider (OpenRouter, OpenAI, etc.)

# Headless CLI (same tools, agents, skill packs) — see docs/cli.md
python -m src.cli -i                          # Textual TUI
python -m src.cli "refactor foo.py"           # one-shot
python -m src.cli --yolo "run the tests"      # auto-approve tool calls
```

You can switch models from the UI dropdown at runtime — no restart needed. If the web UI is already running a model, `python -m src.cli` attaches to it instead of restarting llama-server (sub-second cold start).

## Available models

| Flag | Model | Size | Notes |
|------|-------|------|-------|
| `-m 0.8b` | Qwen 3.5 0.8B Q8_0 | ~1 GB | Fastest, minimal quality |
| `-m 2b` | Qwen 3.5 2B Q4_K_M | ~1.5 GB | Budget GPU |
| `-m 4b` | Qwen 3.5 4B Q4_K_M | ~2.8 GB | Good for 8 GB VRAM |
| `-m 9b` | Qwen 3.5 9B UD-Q4_K_XL | ~6 GB | **Default**, best balance |
| `-m 27b` | Qwen 3.5 27B UD-IQ3_XXS | ~11 GB | Highest quality dense |
| `-m 35b` | Qwen 3.6 35B A3B UD-Q2_K_XL | ~13 GB | MoE, 3B active — fast for its size |
| `-m 35b-agg` | Qwen 3.6 35B Aggressive IQ2_M (uncensored) | ~12 GB | MoE + vision mmproj (HauhauCS) |
| `-m 35b-agg-q4` | Qwen 3.6 35B Aggressive Q4_K_M (uncensored) | ~20 GB | MoE + vision, higher quality quant (HauhauCS) |
| `-m 27b-iq2` | Qwen 3.6 27B UD-IQ2_M | ~11 GB | Dense + vision, local `~/.lmstudio/models/unsloth/Qwen3.6-27B-GGUF/` |
| `-m gemma` | Gemma 4 E4B Aggressive (uncensored) | ~5 GB | Custom — set `MODEL_GEMMA` |
| `-m heretic` | Gemma 4 26B Heretic (uncensored) | ~13 GB | Custom — set `MODEL_HERETIC` + `_CTX` + `_NGL` |
| `-m online` | Any OpenAI-compatible API | 0 GB | Cloud, needs API key |

Most local Qwen models include vision (mmproj). Exceptions: `35b` (Qwen 3.6 A3B Q2 — text-only on the `unsloth` repo); `35b-agg` and `35b-agg-q4` ship their own mmproj. The model manager starts llama-server automatically and downloads models from HuggingFace on first run.

See [docs/setup.md](docs/setup.md) for custom models, per-model overrides, and recommended GPU setups.

## VRAM quick reference

| VRAM | Recommended | Command |
|------|-------------|---------|
| 8 GB | Gemma 4 E4B | `python main.py -m gemma` |
| 16 GB | Gemma 4 26B Heretic | `python main.py -m heretic` |
| 24 GB | Qwen 27B or Qwen 3.6 35B A3B | `python main.py -m 27b` / `-m 35b` |
| No GPU | Gemma 31B free (OpenRouter) | `python main.py -m online2` |

## License

GNU Affero General Public License v3.0 (AGPL-3.0).

AINow is free software: you can use, modify, and redistribute it under the terms of the AGPL-3.0. A key implication: if you run a modified version of AINow over a network (including as a SaaS or internal hosted service), you must make the complete source code of your modified version available to its users. See [LICENSE](LICENSE) for the full terms.

For a commercial license (e.g. to build closed-source products or hosted services without the AGPL source-sharing obligation), contact the author.
