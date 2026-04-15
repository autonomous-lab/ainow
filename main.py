#!/usr/bin/env python3
"""
AINow - Voice Agent Framework

Usage:
    python main.py                      # default: 9B model, server-only
    python main.py --model 9b           # start with 9B model
    python main.py --model 2b           # smaller model

Server mode starts the server and waits for browser connections.
"""

import os
import sys
import signal
import threading
import time
import argparse

import uvicorn
from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that read them
load_dotenv()

from src.server import app
from src.services.model_manager import model_manager, resolve_model_id, MODEL_ALIASES, MODELS
from src.log import setup_logging, Logger, get_logger
import src.server as server_module

# Setup logging
setup_logging()
logger = get_logger("ainow")


def check_environment() -> bool:
    """Check that all required environment variables are set."""
    # At least one LLM provider is needed
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY") and not os.getenv("LLM_BASE_URL"):
        logger.error("Missing environment variable: OPENAI_API_KEY, GROQ_API_KEY, or LLM_BASE_URL")
        return False

    return True


# Max time (seconds) to wait for active calls to finish before forced exit.
DRAIN_TIMEOUT = int(os.getenv("DRAIN_TIMEOUT", "300"))  # 5 minutes default

_uvicorn_server: uvicorn.Server = None


def start_server(port: int) -> None:
    """Start the FastAPI server."""
    global _uvicorn_server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",  # Quiet uvicorn, we have our own logging
    )
    _uvicorn_server = uvicorn.Server(config)
    _uvicorn_server.run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AINow voice agent")
    parser.add_argument(
        "--model", "-m",
        default=None,
        choices=list(MODEL_ALIASES.keys()),
        help="Model to load. If omitted, uses the active agent's last "
             "saved preference (falls back to '9b' if none).",
    )
    args = parser.parse_args()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Resolve the model to load. CLI arg wins, otherwise read the active
    # agent's persisted preference, otherwise fall back to the Qwen 9B default.
    resolved_alias = args.model
    if resolved_alias is None:
        try:
            from src.services import agents as _agent_store
            _agent_store.ensure_default()
            _prefs = _agent_store.read_preferences(_agent_store.get_active())
            pref_model = _prefs.get("model")
            if pref_model and pref_model in MODELS:
                resolved_alias = pref_model
                logger.info(f"Restoring last model from active agent: {pref_model}")
            elif pref_model in MODEL_ALIASES:
                resolved_alias = pref_model
                logger.info(f"Restoring last model from active agent: {pref_model}")
        except Exception as e:
            logger.error(f"Could not read agent model preference: {e}")
    if resolved_alias is None:
        resolved_alias = "9b"

    # Start model (local llama-server or online API)
    model_id = resolve_model_id(resolved_alias)
    model_config = MODELS[model_id]

    if model_config.get("online"):
        # Online mode -- set env vars for LLMService, no llama-server needed
        api_key = os.getenv(model_config["api_key_env"], "")
        if not api_key:
            logger.error(f"Missing {model_config['api_key_env']} in .env")
            sys.exit(1)
        os.environ["LLM_BASE_URL"] = model_config["base_url"]
        os.environ["LLM_API_KEY"] = api_key
        os.environ["LLM_MODEL"] = model_config["model_id"]
        # Don't force BROWSER_STT here — the user controls STT mode via .env.
        # Forcing it would leak into local-model sessions after a model switch.
        logger.info(f"Online mode: {model_config['name']}")
    else:
        os.environ["LLM_MODEL"] = model_id
        # Apply active agent's persisted runtime prefs at boot (vision/ctx/thinking)
        vision_enabled = True
        thinking_enabled = False
        ctx_override = None
        try:
            from src.services import agents as agent_store
            agent_store.ensure_default()
            prefs = agent_store.read_preferences(agent_store.get_active())
            vision_enabled = bool(prefs.get("vision_enabled", True))
            thinking_enabled = bool(prefs.get("thinking_enabled", False))
            if prefs.get("ctx") is not None:
                ctx_override = int(prefs["ctx"])
        except Exception as e:
            logger.error(f"Could not read agent preferences: {e}")
        try:
            model_manager.start(model_id, vision_enabled, ctx_override, thinking_enabled)
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            sys.exit(1)

    # Get port from environment
    port = int(os.getenv("PORT", "3040"))

    # Start server in background thread
    Logger.server_starting(port)
    server_thread = threading.Thread(
        target=start_server,
        args=(port,),
        daemon=True
    )
    server_thread.start()

    # Wait for server to start
    time.sleep(2)
    Logger.server_ready(f"http://localhost:{port}")

    # ── Graceful shutdown on SIGTERM ────────────────────────────────
    def _handle_sigterm(signum, frame):
        """
        Railway (and Docker) send SIGTERM before killing the container.
        We stop accepting new connections and wait for active ones to finish.
        """
        logger.info("SIGTERM received -- starting graceful drain")
        server_module._draining = True

        # If no active calls, exit immediately
        if server_module._active_calls <= 0:
            logger.info("No active calls -- shutting down now")
            if _uvicorn_server:
                _uvicorn_server.should_exit = True
            return

        logger.info(
            f"Waiting up to {DRAIN_TIMEOUT}s for {server_module._active_calls} "
            f"active call(s) to finish..."
        )

        # Poll until calls drain or timeout
        deadline = time.monotonic() + DRAIN_TIMEOUT
        while server_module._active_calls > 0 and time.monotonic() < deadline:
            time.sleep(1)

        remaining = server_module._active_calls
        if remaining > 0:
            logger.warning(f"Drain timeout -- {remaining} call(s) still active, forcing exit")
        else:
            logger.info("All calls drained -- shutting down cleanly")

        if _uvicorn_server:
            _uvicorn_server.should_exit = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        # Server mode -- wait for browser connections
        logger.info("Server mode -- waiting for browser connections (Ctrl+C to end)")

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        Logger.shutdown()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        model_manager.stop()


if __name__ == "__main__":
    main()
