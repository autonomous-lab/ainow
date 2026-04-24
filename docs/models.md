# Available models

## Built-in aliases

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

Most local Qwen models include vision (mmproj). Exceptions: `35b` (Qwen 3.6 A3B Q2 — text-only on the `unsloth` repo); `35b-agg` and `35b-agg-q4` ship their own mmproj. The model manager starts llama-server automatically and downloads models from HuggingFace on first run. When the active model has no mmproj, images in the history are silently stripped and replaced with a marker so llama-server doesn't 500 on a multimodal request.

## VRAM quick reference

| VRAM | Recommended | Command |
|------|-------------|---------|
| 8 GB | Gemma 4 E4B | `python main.py -m gemma` |
| 16 GB | Gemma 4 26B Heretic | `python main.py -m heretic` |
| 24 GB | Qwen 27B or Qwen 3.6 35B A3B | `python main.py -m 27b` / `-m 35b` |
| No GPU | Gemma 31B free (OpenRouter) | `python main.py -m online2` |

## Switching at runtime

- **Web UI** — use the model dropdown at the top-right. Switching reloads llama-server in place; ~3–7s on a cold swap, instant on a no-op.
- **Textual CLI** — `/config` opens a modal (model + vision + thinking + ctx in one apply), or `/model <alias>` for a direct switch. `/model` with no args lists every alias with an active marker.
- **Attach mode** — if the web UI is already running a model, `python -m src.cli` attaches instead of restarting llama-server (sub-second cold start). The CLI reads `/props` from the running server, so its banner reflects the actual context size (e.g. 128K if the web UI set that).

See [setup.md](setup.md) for custom models, per-model ctx/ngl overrides, and the env-var API for registering your own GGUFs.
