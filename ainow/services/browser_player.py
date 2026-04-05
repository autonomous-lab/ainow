"""
Audio player for streaming audio to a browser WebSocket.

Sends raw PCM bytes as binary frames. No drip loop —
the browser handles playback timing via AudioContext.
"""

import json
import base64
import asyncio
from typing import Optional, Callable

from fastapi import WebSocket

from ..log import ServiceLogger

log = ServiceLogger("BrowserPlayer")


class BrowserPlayer:
    """
    Streams audio to a browser client over WebSocket.

    Unlike AudioPlayer (Twilio), this sends raw PCM bytes as binary
    WebSocket frames immediately — no 20ms drip needed because the
    browser's AudioContext handles scheduling.
    """

    def __init__(
        self,
        websocket: WebSocket,
        stream_sid: str,
        on_done: Optional[Callable[[], None]] = None,
    ):
        self._websocket = websocket
        self._stream_sid = stream_sid
        self._on_done = on_done

        self._running = False
        self._tts_done = False
        self._chunk_count = 0

    @property
    def is_playing(self) -> bool:
        return self._running

    async def send_chunk(self, chunk: str) -> None:
        """Decode base64 audio and send raw PCM bytes to browser."""
        if not self._running:
            self._running = True

        try:
            audio_bytes = base64.b64decode(chunk)
            await self._websocket.send_bytes(audio_bytes)
            self._chunk_count += 1
        except Exception as e:
            log.error("Send chunk failed", e)

    def mark_tts_done(self) -> None:
        """Signal that TTS is complete — no more chunks coming."""
        self._tts_done = True
        asyncio.create_task(self._finish())

    async def _finish(self) -> None:
        """Send tts_done message and invoke on_done callback."""
        try:
            await self._websocket.send_text(json.dumps({"type": "tts_done"}))
        except Exception as e:
            log.error("Send tts_done failed", e)

        self._running = False

        if self._on_done:
            self._on_done()

    async def stop_and_clear(self) -> None:
        """Send clear message to browser, reset state."""
        self._running = False
        self._tts_done = False
        self._chunk_count = 0

        try:
            await self._websocket.send_text(json.dumps({"type": "clear"}))
        except Exception as e:
            log.error("Send clear failed", e)

    async def start(self) -> None:
        """Reset state for a new turn (interface compat with AudioPlayer)."""
        self._running = True
        self._tts_done = False
        self._chunk_count = 0

    async def wait_until_done(self) -> None:
        """No-op — browser handles playback timing."""
        pass
