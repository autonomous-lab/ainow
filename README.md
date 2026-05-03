# AINow

**Local-first AI agent framework** — chat, voice, vision, coding, scheduled background tasks, MCP. Runs entirely on your GPU, no cloud APIs required.

## Web UI

![AINow UI](docs/ui-screenshot.png)

```bash
python main.py
# Open http://localhost:3040
```

## Headless CLI

![AINow CLI](docs/cli-screenshot.png)

```bash
python -m src.cli -i
```

## Documentation

| Topic | Docs |
|---|---|
| **Available models** — built-in aliases, VRAM reference, runtime switching | [docs/models.md](docs/models.md) |
| **Headless CLI / TUI** — slash commands, keyboard shortcuts, `@path` expansion | [docs/cli.md](docs/cli.md) |
| **Agents, MCP, scheduled tasks, tools, agentic loop, UI features** | [docs/agents-mcp.md](docs/agents-mcp.md) |
| **Setup, env vars, custom models, security, project structure** | [docs/setup.md](docs/setup.md) |
| **Benchmarks** — Aider Polyglot Python+JS: **100%** local Qwen 3.6 27B IQ2 (83/83) · vs little-coder 45.56% full | [docs/benchmarks.md](docs/benchmarks.md) |

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

## License

GNU Affero General Public License v3.0 (AGPL-3.0).

AINow is free software: you can use, modify, and redistribute it under the terms of the AGPL-3.0. A key implication: if you run a modified version of AINow over a network (including as a SaaS or internal hosted service), you must make the complete source code of your modified version available to its users. See [LICENSE](LICENSE) for the full terms.

For a commercial license (e.g. to build closed-source products or hosted services without the AGPL source-sharing obligation), contact the author.
