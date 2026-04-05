"""
Model manager — starts/stops llama-server with different GGUF models.

Singleton that manages the llama-server process lifecycle.
Started once at server boot via CLI --model flag.

Models are configured via environment variables:
  LLAMA_SERVER_EXE  — path to llama-server binary
  MODELS_DIR        — base directory for GGUF model files
  MODEL_<ALIAS>     — override path for a specific model (e.g. MODEL_GEMMA=/path/to/model.gguf)
"""

import os
import time
import subprocess
from typing import Optional

import httpx

from ..log import get_logger

logger = get_logger("ainow.model_manager")

LLAMA_SERVER_EXE = os.getenv("LLAMA_SERVER_EXE", "llama-server")
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.expanduser("~"), "models"))

# Short aliases for CLI (--model 4b)
MODEL_ALIASES = {
    "0.8b": "qwen3.5-0.8b",
    "2b": "qwen3.5-2b",
    "4b": "qwen3.5-4b",
    "9b": "qwen3.5-9b",
    "27b": "qwen3.5-27b",
    "online": "openrouter-gemini-flash-lite",
}


def _model_path(*parts):
    """Build a model path relative to MODELS_DIR."""
    return os.path.join(MODELS_DIR, *parts)


def _build_models():
    """Build model configs. Env var MODEL_<ALIAS> overrides the default path."""
    models = {
        "qwen3.5-0.8b": {
            "name": "Qwen 0.8B",
            "model": _model_path("Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q8_0.gguf"),
            "mmproj": _model_path("Qwen3.5-0.8B-GGUF", "mmproj-Qwen3.5-0.8B-BF16.gguf"),
        },
        "qwen3.5-2b": {
            "name": "Qwen 2B",
            "model": _model_path("Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf"),
            "mmproj": _model_path("Qwen3.5-2B-GGUF", "mmproj-Qwen3.5-2B-BF16.gguf"),
        },
        "qwen3.5-4b": {
            "name": "Qwen 4B",
            "model": _model_path("Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
            "mmproj": _model_path("Qwen3.5-4B-GGUF", "mmproj-Qwen3.5-4B-BF16.gguf"),
        },
        "qwen3.5-9b": {
            "name": "Qwen 9B",
            "model": _model_path("Qwen3.5-9B-GGUF", "Qwen3.5-9B-UD-Q4_K_XL.gguf"),
            "mmproj": _model_path("Qwen3.5-9B-GGUF", "mmproj-Qwen3.5-9B-BF16.gguf"),
        },
        "qwen3.5-27b": {
            "name": "Qwen 27B",
            "model": _model_path("Qwen3.5-27B-GGUF", "Qwen3.5-27B-UD-IQ3_XXS.gguf"),
            "mmproj": _model_path("Qwen3.5-27B-GGUF", "mmproj-BF16.gguf"),
            "ctx": "32768",
        },
        "openrouter-gemini-flash-lite": {
            "name": "Gemini Flash Lite (OpenRouter)",
            "online": True,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "model_id": "google/gemini-3.1-flash-lite-preview",
        },
    }

    # Register extra models from env vars: MODEL_GEMMA=/path/to/dir
    # Format: MODEL_<ALIAS>=<model_gguf_path>;<mmproj_gguf_path>;<display_name>
    # Or:     MODEL_<ALIAS>=<directory>  (auto-detect .gguf and mmproj files)
    for key, value in os.environ.items():
        if not key.startswith("MODEL_") or key in ("MODELS_DIR", "MODEL_ALIASES"):
            continue
        alias = key[6:].lower()  # MODEL_GEMMA -> gemma
        if alias in MODEL_ALIASES or alias in models:
            continue

        if ";" in value:
            # Explicit: MODEL_GEMMA=/path/model.gguf;/path/mmproj.gguf;Gemma 4B
            parts = value.split(";")
            model_file = parts[0].strip()
            mmproj_file = parts[1].strip() if len(parts) > 1 else ""
            display_name = parts[2].strip() if len(parts) > 2 else alias.upper()
        elif os.path.isdir(value):
            # Directory: auto-detect model and mmproj files
            model_file = ""
            mmproj_file = ""
            display_name = alias.upper()
            for f in os.listdir(value):
                fl = f.lower()
                if fl.endswith(".gguf"):
                    if "mmproj" in fl:
                        mmproj_file = os.path.join(value, f)
                    elif not model_file:
                        model_file = os.path.join(value, f)
        else:
            # Single file path — assume it's the model, no mmproj
            model_file = value
            mmproj_file = ""
            display_name = alias.upper()

        if model_file:
            MODEL_ALIASES[alias] = alias
            config = {"name": display_name, "model": model_file}
            if mmproj_file:
                config["mmproj"] = mmproj_file
            models[alias] = config
            logger.info(f"Registered custom model '{alias}' from env")

    return models


MODELS = _build_models()

COMMON_ARGS = [
    "-c", "262144",
    "-np", "1",
    "--fit", "on",
    "--fit-target", "1024",
    "-fa", "on",
    "-t", "20",
    "--no-mmap",
    "--jinja",
    "-ctk", "q4_0",
    "-ctv", "q4_0",
    "--reasoning", "off",
    "--reasoning-budget", "0",
    "--port", str(LLAMA_SERVER_PORT),
]


def resolve_model_id(name: str) -> str:
    """Resolve a short alias (e.g. '4b') or full id to a full model id."""
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    if name in MODELS:
        return name
    raise ValueError(f"Unknown model: {name}. Valid: {', '.join(MODEL_ALIASES.keys())}")


class ModelManager:
    """Singleton managing the llama-server process."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._current_model: Optional[str] = None

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    def _check_health_sync(self) -> bool:
        """Synchronous health check."""
        try:
            r = httpx.get(
                f"http://localhost:{LLAMA_SERVER_PORT}/health",
                timeout=2.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _stop_server_sync(self) -> None:
        """Stop the current llama-server process (synchronous)."""
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                time.sleep(2)
                if self._process.poll() is None:
                    self._process.kill()
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
            self._process = None
        else:
            # Try to kill whatever is on the port
            try:
                result = subprocess.run(
                    ["netstat", "-ano"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace", timeout=5,
                )
                for line in result.stdout.splitlines():
                    if f":{LLAMA_SERVER_PORT}" in line and "LISTENING" in line:
                        parts = line.split()
                        pid = int(parts[-1])
                        subprocess.run(
                            ["taskkill", "/PID", str(pid), "/F"],
                            capture_output=True, timeout=5,
                        )
                        logger.info(f"Killed existing process on port {LLAMA_SERVER_PORT} (PID {pid})")
                        time.sleep(1)
                        break
            except Exception as e:
                logger.error(f"Error killing existing process: {e}")

        self._current_model = None

    def start(self, model_id: str) -> None:
        """
        Start llama-server with the given model (synchronous, blocking).

        Called once at server boot before the event loop starts.
        Kills any existing llama-server first.
        """
        if model_id not in MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        config = MODELS[model_id]
        logger.info(f"Starting llama-server with {config['name']}...")

        # Stop any existing server
        if self._check_health_sync():
            logger.info("Stopping existing llama-server...")
            self._stop_server_sync()

        # Start new server
        cmd = [LLAMA_SERVER_EXE, "-m", config["model"]]
        if config.get("mmproj"):
            cmd += ["--mmproj", config["mmproj"]]
        cmd += COMMON_ARGS
        # Per-model context override
        if "ctx" in config:
            cmd[cmd.index("-c") + 1] = config["ctx"]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

        # Wait for health check
        max_wait = 120
        poll_interval = 1.0
        elapsed = 0.0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited unexpectedly (code {self._process.returncode})")

            if self._check_health_sync():
                self._current_model = model_id
                logger.info(f"llama-server ready with {config['name']} ({elapsed:.0f}s)")
                return

            if int(elapsed) % 10 == 0:
                logger.info(f"  Waiting for llama-server... ({elapsed:.0f}s)")

        raise TimeoutError(f"llama-server did not become healthy within {max_wait}s")

    def stop(self) -> None:
        """Stop llama-server (synchronous)."""
        self._stop_server_sync()


# Singleton
model_manager = ModelManager()
