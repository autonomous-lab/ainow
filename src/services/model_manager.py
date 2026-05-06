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

# Vision on/off and preferred model are now stored per-agent in meta.json
# under `preferences`. See services/agents.py.

# Short aliases for CLI (--model 4b)
MODEL_ALIASES = {
    "0.8b": "qwen3.5-0.8b",
    "2b": "qwen3.5-2b",
    "4b": "qwen3.5-4b",
    "9b": "qwen3.5-9b",
    "27b": "qwen3.5-27b",
    "35b": "qwen3.6-35b",
    "35b-agg": "qwen3.6-35b-agg",
    "35b-agg-q4": "qwen3.6-35b-agg-q4",
    "27b-iq2": "qwen3.6-27b-iq2",
    "online": "online",
    "online2": "online2",
    "deepseek": "deepseek-v4-flash",
}


def _model_path(*parts):
    """Build a model path relative to MODELS_DIR."""
    return os.path.join(MODELS_DIR, *parts)


def _build_models():
    """Build model configs. Env var MODEL_<ALIAS> overrides the default path."""
    models = {
        "qwen3.5-0.8b": {
            "name": "Qwen 0.8B",
            "hf_repo": "lmstudio-community/Qwen3.5-0.8B-GGUF",
            "hf_files": ["Qwen3.5-0.8B-Q8_0.gguf", "mmproj-Qwen3.5-0.8B-BF16.gguf"],
            "model": _model_path("Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q8_0.gguf"),
            "mmproj": _model_path("Qwen3.5-0.8B-GGUF", "mmproj-Qwen3.5-0.8B-BF16.gguf"),
        },
        "qwen3.5-2b": {
            "name": "Qwen 2B",
            "hf_repo": "lmstudio-community/Qwen3.5-2B-GGUF",
            "hf_files": ["Qwen3.5-2B-Q4_K_M.gguf", "mmproj-Qwen3.5-2B-BF16.gguf"],
            "model": _model_path("Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf"),
            "mmproj": _model_path("Qwen3.5-2B-GGUF", "mmproj-Qwen3.5-2B-BF16.gguf"),
        },
        "qwen3.5-4b": {
            "name": "Qwen 4B",
            "hf_repo": "lmstudio-community/Qwen3.5-4B-GGUF",
            "hf_files": ["Qwen3.5-4B-Q4_K_M.gguf", "mmproj-Qwen3.5-4B-BF16.gguf"],
            "model": _model_path("Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
            "mmproj": _model_path("Qwen3.5-4B-GGUF", "mmproj-Qwen3.5-4B-BF16.gguf"),
        },
        "qwen3.5-9b": {
            "name": "Qwen 9B",
            "hf_repo": "lmstudio-community/Qwen3.5-9B-GGUF",
            "hf_files": ["Qwen3.5-9B-UD-Q4_K_XL.gguf", "mmproj-Qwen3.5-9B-BF16.gguf"],
            "model": _model_path("Qwen3.5-9B-GGUF", "Qwen3.5-9B-UD-Q4_K_XL.gguf"),
            "mmproj": _model_path("Qwen3.5-9B-GGUF", "mmproj-Qwen3.5-9B-BF16.gguf"),
        },
        "qwen3.5-27b": {
            "name": "Qwen 27B",
            "hf_repo": "lmstudio-community/Qwen3.5-27B-GGUF",
            "hf_files": ["Qwen3.5-27B-UD-IQ3_XXS.gguf", "mmproj-BF16.gguf"],
            "model": _model_path("Qwen3.5-27B-GGUF", "Qwen3.5-27B-UD-IQ3_XXS.gguf"),
            "mmproj": _model_path("Qwen3.5-27B-GGUF", "mmproj-BF16.gguf"),
            "ctx": "32768",
        },
        "qwen3.6-35b": {
            "name": "Qwen 3.6 35B A3B (Q2)",
            "hf_repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "hf_files": ["Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"],
            "model": _model_path("Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
            "ctx": "32768",
        },
        "qwen3.6-35b-agg": {
            "name": "Qwen 3.6 35B Aggressive (uncensored, IQ2)",
            "hf_repo": "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
            "hf_files": [
                "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf",
                "mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf",
            ],
            "model": _model_path("Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive", "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf"),
            "mmproj": _model_path("Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive", "mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"),
            "ctx": "32768",
        },
        "qwen3.6-35b-agg-q4": {
            "name": "Qwen 3.6 35B Aggressive (uncensored, Q4)",
            "model": os.path.join(os.path.expanduser("~"), ".lmstudio", "models", "HauhauCS", "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive", "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"),
            "mmproj": os.path.join(os.path.expanduser("~"), ".lmstudio", "models", "HauhauCS", "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive", "mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"),
            "ctx": "32768",
        },
        "qwen3.6-27b-iq2": {
            "name": "Qwen 3.6 27B (IQ2)",
            "model": os.path.join(os.path.expanduser("~"), ".lmstudio", "models", "unsloth", "Qwen3.6-27B-GGUF", "Qwen3.6-27B-UD-IQ2_M.gguf"),
            "mmproj": os.path.join(os.path.expanduser("~"), ".lmstudio", "models", "unsloth", "Qwen3.6-27B-GGUF", "mmproj-F32.gguf"),
            "ctx": "32768",
        },
        "online": {
            "name": os.getenv("ONLINE_MODEL", "google/gemini-3.1-flash-lite-preview"),
            "online": True,
            "base_url": os.getenv("ONLINE_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key_env": "ONLINE_API_KEY",
            "model_id": os.getenv("ONLINE_MODEL", "google/gemini-3.1-flash-lite-preview"),
        },
        "online2": {
            "name": os.getenv("ONLINE2_MODEL", "google/gemma-4-31b-it:free"),
            "online": True,
            "base_url": os.getenv("ONLINE2_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key_env": "ONLINE_API_KEY",  # shares the same OpenRouter key
            "model_id": os.getenv("ONLINE2_MODEL", "google/gemma-4-31b-it:free"),
        },
        "deepseek-v4-flash": {
            "name": "DeepSeek V4 Flash (online)",
            "online": True,
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key_env": "ONLINE_API_KEY",  # shares the same OpenRouter key
            "model_id": os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-v4-flash"),
        },
    }

    # Register extra models from env vars: MODEL_GEMMA=/path/to/dir
    # Format: MODEL_<ALIAS>=<model_gguf_path>;<mmproj_gguf_path>;<display_name>
    # Or:     MODEL_<ALIAS>=<directory>  (auto-detect .gguf and mmproj files)
    for key, value in os.environ.items():
        if not key.startswith("MODEL_") or key in ("MODELS_DIR", "MODEL_ALIASES"):
            continue
        # Skip per-model override env vars (handled below per-alias)
        if key.endswith(("_CTX", "_NGL")):
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
            # Optional per-model overrides via MODEL_<ALIAS>_CTX / _NGL env vars
            ctx_override = os.environ.get(f"MODEL_{alias.upper()}_CTX")
            if ctx_override:
                config["ctx"] = ctx_override
            ngl_override = os.environ.get(f"MODEL_{alias.upper()}_NGL")
            if ngl_override:
                config["ngl"] = ngl_override
            models[alias] = config
            logger.info(f"Registered custom model '{alias}' from env")

    return models


MODELS = _build_models()

# KV cache quantization type — TurboQuant provides turbo2/turbo3/turbo4
# for 3.8-6.4x compression. Falls back to q4_0 on mainline llama-server.
KV_CACHE_TYPE = os.getenv("KV_CACHE_TYPE", "turbo3")

def _should_load_mmproj(config: dict, vision_enabled: bool) -> bool:
    """Load mmproj only when needed for vision or audio-LLM input."""
    return bool(config.get("mmproj")) and (vision_enabled or bool(os.getenv("AUDIO_LLM")))


COMMON_ARGS = [
    "-c", "262144",
    "-np", "1",
    "--fit", "on",
    "--fit-target", "1024",
    "-fa", "on",
    "-t", "20",
    "-b", "1024",
    "-ub", "1024",
    "--no-mmap",
    "--jinja",
    "-ctk", KV_CACHE_TYPE,
    "-ctv", KV_CACHE_TYPE,
    "--port", str(LLAMA_SERVER_PORT),
    # NOTE: --reasoning flags are added dynamically in start() based on
    # the thinking_enabled parameter (per-agent preference).
]


def _try_build_turboquant(install_dir: str) -> bool:
    """Attempt to build llama-server from the TurboQuant fork.

    Returns True if build succeeded and the binary exists.
    """
    import platform
    import shutil

    repo_url = "https://github.com/TheTom/llama-cpp-turboquant.git"
    branch = "feature/turboquant-kv-cache"
    build_dir = os.path.join(os.path.dirname(install_dir), ".llama-build")

    # Check prerequisites
    if not shutil.which("cmake"):
        logger.info("cmake not found, skipping TurboQuant build")
        return False
    if not shutil.which("git"):
        logger.info("git not found, skipping TurboQuant build")
        return False

    try:
        # Clone
        if os.path.isdir(build_dir):
            logger.info("Updating TurboQuant clone...")
            subprocess.run(["git", "fetch", "origin"], cwd=build_dir, timeout=120,
                           capture_output=True)
            subprocess.run(["git", "checkout", branch], cwd=build_dir, timeout=30,
                           capture_output=True)
            subprocess.run(["git", "reset", "--hard", f"origin/{branch}"],
                           cwd=build_dir, timeout=30, capture_output=True)
        else:
            logger.info(f"Cloning TurboQuant fork ({branch})...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, repo_url, build_dir],
                timeout=300, check=True, capture_output=True,
            )

        # Configure
        cmake_args = [
            "cmake", "-B", "build",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        if shutil.which("nvcc"):
            cmake_args.append("-DGGML_CUDA=ON")
            logger.info("Building TurboQuant with CUDA support")
        else:
            logger.info("Building TurboQuant (CPU only — install CUDA for GPU)")

        subprocess.run(cmake_args, cwd=build_dir, timeout=120, check=True,
                       capture_output=True)

        # Build
        logger.info("Compiling TurboQuant llama-server (this may take a few minutes)...")
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "-j",
             str(os.cpu_count() or 4)],
            cwd=build_dir, timeout=600, check=True, capture_output=True,
        )

        # Find binary
        system = platform.system().lower()
        exe_name = "llama-server.exe" if system == "windows" else "llama-server"
        search_dirs = [
            os.path.join(build_dir, "build", "bin", "Release"),
            os.path.join(build_dir, "build", "bin"),
            os.path.join(build_dir, "build", "Release", "bin"),
        ]
        for d in search_dirs:
            candidate = os.path.join(d, exe_name)
            if os.path.isfile(candidate):
                os.makedirs(install_dir, exist_ok=True)
                import shutil as _sh
                _sh.copy2(candidate, os.path.join(install_dir, exe_name))
                # Copy runtime libs (DLLs / .so)
                ext = "*.dll" if system == "windows" else "*.so"
                import glob
                for lib in glob.glob(os.path.join(d, ext)):
                    _sh.copy2(lib, install_dir)
                logger.info(f"TurboQuant llama-server built and installed to {install_dir}")
                return True

        logger.error("TurboQuant build completed but binary not found")
        return False

    except subprocess.TimeoutExpired:
        logger.error("TurboQuant build timed out")
        return False
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")[-500:] if e.stderr else ""
        logger.error(f"TurboQuant build failed: {stderr}")
        return False
    except Exception as e:
        logger.error(f"TurboQuant build error: {e}")
        return False


def _ensure_llama_server():
    """Ensure llama-server binary is available.

    Priority:
    1. User-specified path via LLAMA_SERVER_EXE env var
    2. Build from TurboQuant fork (if cmake + git available)
    3. Download mainline llama.cpp release (fallback)
    """
    global LLAMA_SERVER_EXE

    if os.path.isfile(LLAMA_SERVER_EXE):
        return

    # Only auto-download if using default name (not a custom path)
    if os.path.dirname(LLAMA_SERVER_EXE):
        # User specified a full path that doesn't exist
        raise FileNotFoundError(f"llama-server not found at: {LLAMA_SERVER_EXE}")

    import platform
    import zipfile

    system = platform.system().lower()
    exe_name = "llama-server.exe" if system == "windows" else "llama-server"
    install_dir = os.path.join(os.path.dirname(MODELS_DIR), "llama-server")

    # Try 1: Build from TurboQuant fork (best quality — KV cache compression)
    logger.info("llama-server not found, attempting TurboQuant build...")
    if _try_build_turboquant(install_dir):
        LLAMA_SERVER_EXE = os.path.join(install_dir, exe_name)
        if os.path.isfile(LLAMA_SERVER_EXE):
            return

    # Try 2: Download mainline release (fallback)
    logger.info("TurboQuant build unavailable, downloading mainline llama.cpp release...")

    resp = httpx.get(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
        timeout=30,
    )
    resp.raise_for_status()
    release = resp.json()
    tag = release["tag_name"]

    # Find the right asset
    target = None
    cudart_target = None
    for asset in release["assets"]:
        name = asset["name"]
        if system == "windows" and "win" in name and "cuda" in name and name.endswith(".zip"):
            if "cudart" not in name:
                target = asset
            else:
                cudart_target = asset
        elif system == "linux" and "linux" in name and "cuda" in name and name.endswith(".zip"):
            if "cudart" not in name:
                target = asset
            else:
                cudart_target = asset

    if not target:
        raise RuntimeError(f"No suitable llama-server release found for {system}")

    os.makedirs(install_dir, exist_ok=True)

    for asset in [target, cudart_target]:
        if not asset:
            continue
        url = asset["browser_download_url"]
        zip_path = os.path.join(install_dir, asset["name"])

        logger.info(f"Downloading {asset['name']}...")
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_bytes(8192):
                    f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(install_dir)
        os.remove(zip_path)

    LLAMA_SERVER_EXE = os.path.join(install_dir, exe_name)

    if not os.path.isfile(LLAMA_SERVER_EXE):
        raise RuntimeError(f"Download succeeded but {exe_name} not found in {install_dir}")

    logger.info(f"llama-server {tag} installed to {install_dir}")


def _ensure_model_files(config):
    """Download model files from HuggingFace if not present."""
    if "hf_repo" not in config:
        return  # Custom or online model, user manages files

    from huggingface_hub import hf_hub_download

    for filename in config.get("hf_files", []):
        # The subfolder in MODELS_DIR is derived from the HF repo name's last part
        subfolder = config["hf_repo"].split("/")[-1]
        local_path = os.path.join(MODELS_DIR, subfolder, filename)

        if os.path.exists(local_path):
            continue

        logger.info(f"Downloading {filename} from {config['hf_repo']}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        hf_hub_download(
            repo_id=config["hf_repo"],
            filename=filename,
            local_dir=os.path.join(MODELS_DIR, subfolder),
        )
        logger.info(f"Downloaded {filename}")


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
        self._last_vision_enabled: bool = True
        self._last_ctx: int = 0
        self._last_thinking_enabled: bool = False

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    @property
    def vision_enabled(self) -> bool:
        """Whether the currently-running llama-server was started with mmproj."""
        return self._last_vision_enabled

    @property
    def context_size(self) -> int:
        """Effective -c the current server was started with (0 if not running)."""
        return self._last_ctx

    @property
    def thinking_enabled(self) -> bool:
        """Whether the current server was started with --reasoning on."""
        return self._last_thinking_enabled

    def get_context_size(self) -> int:
        """Return the effective context length for the active model.
        Uses the runtime value (after user overrides via UI), not the config default."""
        if not self._current_model:
            return 0
        config = MODELS.get(self._current_model, {})
        if config.get("online"):
            return 0
        # _last_ctx is set during start() with the actual -c value used
        if self._last_ctx > 0:
            return self._last_ctx
        return 0

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

    def start(
        self,
        model_id: str,
        vision_enabled: bool = True,
        ctx_override: Optional[int] = None,
        thinking_enabled: bool = False,
    ) -> None:
        """
        Start llama-server with the given model (synchronous, blocking).

        Called once at server boot before the event loop starts.
        Kills any existing llama-server first.

        vision_enabled: if False, the model's --mmproj is NOT loaded even if
          it exists in the config. Used to save VRAM.
        ctx_override: if provided, overrides both the global `-c` flag and
          any per-model MODEL_<NAME>_CTX env var. Wins over everything.
        """
        if model_id not in MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        config = MODELS[model_id]

        # Compute what the effective ctx would be for this request.
        if ctx_override is not None:
            _req_ctx = int(ctx_override)
        elif "ctx" in config:
            _req_ctx = int(config["ctx"])
        else:
            try:
                _req_ctx = int(COMMON_ARGS[COMMON_ARGS.index("-c") + 1])
            except Exception:
                _req_ctx = 0

        # Short-circuit: if the same model is already running with the same
        # params and the server is healthy, skip the ~3-7s unload/reload cycle.
        if (
            self._current_model == model_id
            and self._last_vision_enabled == vision_enabled
            and self._last_thinking_enabled == thinking_enabled
            and self._last_ctx == _req_ctx
            and self._check_health_sync()
        ):
            logger.info(
                f"llama-server already running with {config['name']} "
                f"(ctx={_req_ctx}, vision={vision_enabled}, thinking={thinking_enabled}) — reusing"
            )
            return

        self._last_vision_enabled = vision_enabled
        self._last_thinking_enabled = thinking_enabled

        # Auto-download llama-server and model files if missing
        if not config.get("online"):
            _ensure_llama_server()
            _ensure_model_files(config)

        logger.info(f"Starting llama-server with {config['name']}...")

        # Stop any existing server
        if self._check_health_sync():
            logger.info("Stopping existing llama-server...")
            self._stop_server_sync()

        # Start new server
        cmd = [LLAMA_SERVER_EXE, "-m", config["model"]]
        # Load mmproj only when vision is enabled or audio-LLM mode needs it.
        if _should_load_mmproj(config, vision_enabled):
            cmd += ["--mmproj", config["mmproj"]]
        cmd += COMMON_ARGS
        # Context size priority: explicit override > per-model env > global default
        if ctx_override is not None:
            cmd[cmd.index("-c") + 1] = str(int(ctx_override))
            effective_ctx = int(ctx_override)
        elif "ctx" in config:
            cmd[cmd.index("-c") + 1] = config["ctx"]
            effective_ctx = int(config["ctx"])
        else:
            try:
                effective_ctx = int(cmd[cmd.index("-c") + 1])
            except Exception:
                effective_ctx = 0
        self._last_ctx = effective_ctx
        # Per-model GPU layer offload override (-ngl)
        if "ngl" in config:
            cmd += ["-ngl", str(config["ngl"])]
        # Reasoning / thinking mode
        if thinking_enabled:
            cmd += ["--reasoning", "on"]
        else:
            # Must explicitly disable thinking for Qwen 3.5 models whose Jinja
            # template enables it by default. Both flags are needed:
            # --reasoning off  = sets enable_thinking=false in the template
            # --reasoning-budget 0 = suppresses any residual <think> blocks
            cmd += ["--reasoning", "off", "--reasoning-budget", "0"]

        # MTP (Multi-Token Prediction) speculative decoding
        # See https://github.com/ggml-org/llama.cpp/pull/22673 — adds ~2x
        # decode speedup on Qwen 3.5/3.6 by loading pre-trained MTP heads
        # from the same GGUF and emitting N draft tokens per forward pass.
        # Opt-in (the PR is still in draft + needs a binary built from the
        # branch). Auto-skip if the model isn't a known MTP-capable family.
        if os.getenv("AINOW_MTP", "").strip() in ("1", "true", "yes"):
            _model_path = config.get("model", "").lower()
            _is_mtp_family = any(tag in _model_path for tag in ("qwen3.5", "qwen3.6"))
            if _is_mtp_family:
                _draft_n = os.getenv("AINOW_MTP_DRAFT_N", "3").strip() or "3"
                cmd += ["--spec-type", "mtp", "--spec-draft-n-max", _draft_n]
                logger.info(f"MTP speculative decoding enabled (--spec-draft-n-max {_draft_n})")
            else:
                logger.info("AINOW_MTP set but current model isn't Qwen 3.5/3.6 — flag ignored")

        # Try to start — if TurboQuant cache types fail (mainline binary),
        # fall back to standard q4_0.
        _attempts = [(cmd, KV_CACHE_TYPE)]
        if KV_CACHE_TYPE.startswith("turbo"):
            fallback_cmd = list(cmd)
            try:
                ki = fallback_cmd.index("-ctk")
                fallback_cmd[ki + 1] = "q4_0"
                vi = fallback_cmd.index("-ctv")
                fallback_cmd[vi + 1] = "q4_0"
                _attempts.append((fallback_cmd, "q4_0"))
            except ValueError:
                pass

        for _cmd, _cache_type in _attempts:
            logger.info(f"Starting llama-server (KV cache: {_cache_type})...")
            self._process = subprocess.Popen(
                _cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )

            # Wait for health check
            max_wait = 120
            poll_interval = 1.0
            elapsed = 0.0
            crashed = False

            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval

                if self._process.poll() is not None:
                    # Server crashed — try fallback if available
                    if _cache_type != "q4_0" and len(_attempts) > 1:
                        logger.info(f"llama-server crashed with {_cache_type} cache, trying q4_0 fallback...")
                        crashed = True
                        break
                    raise RuntimeError(f"llama-server exited unexpectedly (code {self._process.returncode})")

                if self._check_health_sync():
                    self._current_model = model_id
                    if _cache_type.startswith("turbo"):
                        logger.info(f"llama-server ready with {config['name']} + TurboQuant {_cache_type} ({elapsed:.0f}s)")
                    else:
                        logger.info(f"llama-server ready with {config['name']} ({elapsed:.0f}s)")
                    return

                if int(elapsed) % 10 == 0:
                    logger.info(f"  Waiting for llama-server... ({elapsed:.0f}s)")

            if not crashed:
                raise TimeoutError(f"llama-server did not become healthy within {max_wait}s")

    def stop(self) -> None:
        """Stop llama-server (synchronous)."""
        self._stop_server_sync()


# Singleton
model_manager = ModelManager()
