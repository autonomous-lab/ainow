#!/usr/bin/env python3
"""
AINow - Voice Agent Framework

Usage:
    python main.py                      # default: 4B model, server-only
    python main.py --model 9b           # start with 9B model
    python main.py +1234567890          # outbound call mode
    python main.py --model 2b +1234567890

Server-only mode starts the server and waits for inbound calls.
Outbound mode additionally initiates a call to the specified number.
"""

import os
import sys
import signal
import threading
import time
import argparse

import uvicorn
from dotenv import load_dotenv

from ainow.server import app
from ainow.services.twilio_client import make_outbound_call
from ainow.services.model_manager import model_manager, resolve_model_id, MODEL_ALIASES, MODELS
from ainow.log import setup_logging, Logger, get_logger
import ainow.server as server_module

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger("ainow")


def check_environment(require_twilio: bool = False) -> bool:
    """Check that all required environment variables are set."""
    required_vars = []

    # External APIs only required if using server TTS without local alternative
    if os.getenv("SERVER_TTS") and not os.getenv("LOCAL_TTS_VOICE"):
        required_vars.append("ELEVENLABS_API_KEY")
    if not os.getenv("LOCAL_TTS_VOICE"):
        # Deepgram only needed if not using browser STT
        if os.getenv("DEEPGRAM_API_KEY"):
            pass  # available, will be used
        # else: browser STT mode, no deepgram needed

    # At least one LLM provider is needed
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY") and not os.getenv("LLM_BASE_URL"):
        logger.error("Missing environment variable: OPENAI_API_KEY, GROQ_API_KEY, or LLM_BASE_URL")
        return False

    if require_twilio:
        required_vars += [
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_PHONE_NUMBER",
            "TWILIO_PUBLIC_URL",
        ]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
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
        default="9b",
        choices=list(MODEL_ALIASES.keys()),
        help="Model size to load (default: 9b)",
    )
    parser.add_argument(
        "phone_number",
        nargs="?",
        default=None,
        help="Phone number for outbound call (e.g. +1234567890)",
    )
    args = parser.parse_args()

    phone_number = args.phone_number
    if phone_number and not phone_number.startswith("+"):
        print("Error: Phone number must start with +")
        sys.exit(1)

    # Check environment (Twilio vars only needed for outbound calls)
    if not check_environment(require_twilio=phone_number is not None):
        sys.exit(1)

    # Start model (local llama-server or online API)
    model_id = resolve_model_id(args.model)
    model_config = MODELS[model_id]

    if model_config.get("online"):
        # Online mode — set env vars for LLMService, no llama-server needed
        api_key = os.getenv(model_config["api_key_env"], "")
        if not api_key:
            logger.error(f"Missing {model_config['api_key_env']} in .env")
            sys.exit(1)
        os.environ["LLM_BASE_URL"] = model_config["base_url"]
        os.environ["LLM_API_KEY"] = api_key
        os.environ["LLM_MODEL"] = model_config["model_id"]
        os.environ["BROWSER_STT"] = "1"  # Use browser speech recognition (no Whisper needed)
        logger.info(f"Online mode: {model_config['name']}")
    else:
        os.environ["LLM_MODEL"] = model_id
        try:
            model_manager.start(model_id)
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            sys.exit(1)

    # Get port from environment
    port = int(os.getenv("PORT", "3040"))
    public_url = os.getenv("TWILIO_PUBLIC_URL", "")

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
    Logger.server_ready(public_url)
    
    # ── Graceful shutdown on SIGTERM ────────────────────────────────
    def _handle_sigterm(signum, frame):
        """
        Railway (and Docker) send SIGTERM before killing the container.
        We stop accepting new calls and wait for active ones to finish.
        """
        logger.info("SIGTERM received — starting graceful drain")
        server_module._draining = True

        # If no active calls, exit immediately
        if server_module._active_calls <= 0:
            logger.info("No active calls — shutting down now")
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
            logger.warning(f"Drain timeout — {remaining} call(s) still active, forcing exit")
        else:
            logger.info("All calls drained — shutting down cleanly")

        if _uvicorn_server:
            _uvicorn_server.should_exit = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        if phone_number:
            # Outbound call mode
            Logger.call_initiating(phone_number)
            call_sid = make_outbound_call(phone_number)
            Logger.call_initiated(call_sid)
            logger.info("Waiting for call to connect... (Ctrl+C to end)")
        else:
            # Server-only mode — wait for inbound calls
            logger.info("Server-only mode — waiting for inbound calls (Ctrl+C to end)")

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
