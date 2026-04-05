"""
Agent -- self-contained LLM -> TTS -> Player pipeline.

Encapsulates the entire agent response lifecycle.
Owns conversation history across turns.

    start_turn(transcript) -> add to history -> LLM -> TTS -> Player -> Twilio
    cancel_turn()          -> cancel all, keep history

TTS connections are managed by TTSPool (see services/tts_pool.py).
"""

import asyncio
import time
from typing import Optional, Callable, Awaitable, List, Dict, Any

from fastapi import WebSocket

from .services.llm import LLMService
from .services.tts import TTSService
from .services.tts_pool import TTSPool
from .services.player import AudioPlayer
from .services.tools import TOOLS
from .tracer import Tracer
from .log import ServiceLogger

log = ServiceLogger("Agent")


def _ms_since(t0: float) -> int:
    """Milliseconds elapsed since t0."""
    return int((time.monotonic() - t0) * 1000)


class Agent:
    """
    Self-contained agent response pipeline.

    LLM is persistent (keeps conversation history across turns).
    TTS connections come from TTSPool (pre-connected, with TTL eviction).
    Player is created fresh per turn.
    """

    def __init__(
        self,
        websocket: WebSocket,
        stream_sid: str,
        on_done: Callable[[], None],
        tts_pool: Optional[TTSPool],
        tracer: Tracer,
        player_cls: type = AudioPlayer,
        on_response_token: Optional[Callable[[str], Awaitable[None]]] = None,
        on_pipeline: Optional[Callable[[str, str], Awaitable[None]]] = None,
        on_tool_call: Optional[Callable[[str, Any], Awaitable[None]]] = None,
        on_tool_result: Optional[Callable[[str, str, str], Awaitable[None]]] = None,
        tts_muted_fn: Optional[Callable[[], bool]] = None,
        system_prompt: Optional[str] = None,
        on_tool_confirm: Optional[Callable[[str, Any], Awaitable[bool]]] = None,
        on_browser_tool: Optional[Callable[[str, dict], Awaitable[str]]] = None,
    ):
        self._websocket = websocket
        self._stream_sid = stream_sid
        self._on_done = on_done
        self._tts_pool = tts_pool
        self._tracer = tracer
        self._player_cls = player_cls
        self._on_response_token = on_response_token
        self._on_pipeline = on_pipeline
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result
        self._tts_muted_fn = tts_muted_fn or (lambda: False)

        self._external_on_tool_call = on_tool_call

        # Persistent LLM -- keeps conversation history across turns
        self._llm = LLMService(
            on_token=self._on_llm_token,
            on_done=self._on_llm_done,
            tools=TOOLS,
            on_tool_call=self._on_tool_call_wrapper,
            on_tool_result=on_tool_result,
            system_prompt=system_prompt,
            on_tool_confirm=on_tool_confirm,
            on_browser_tool=on_browser_tool,
        )

        # Active per-turn services (set during start, cleared on cancel)
        self._tts: Optional[TTSService] = None
        self._player: Optional[AudioPlayer] = None
        self._active = False

        # Current turn number (for tracer)
        self._turn: int = 0

        # Latency milestones (monotonic timestamps, reset each turn)
        self._t0: float = 0.0
        self._t_tts_conn: float = 0.0
        self._t_first_token: float = 0.0
        self._t_first_audio: float = 0.0
        self._got_first_token = False
        self._got_first_audio = False

    @property
    def is_turn_active(self) -> bool:
        return self._active

    @property
    def model(self) -> str:
        return self._llm.model

    @model.setter
    def model(self, value: str) -> None:
        self._llm.model = value

    def set_system_prompt(self, prompt: str) -> None:
        self._llm.system_prompt = prompt

    def inject_image(self, data_url: str) -> None:
        """Inject a captured image into the LLM conversation history."""
        self._llm.inject_image(data_url)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._llm.clear_history()

    def restore_history(self, messages: list) -> None:
        """Restore conversation history from client-side messages."""
        self._llm.restore_history(messages)

    @property
    def history(self) -> List[Dict[str, str]]:
        """Read-only access to conversation history (owned by LLM)."""
        return self._llm.history

    async def _pipeline(self, stage: str, status: str, **extra) -> None:
        if self._on_pipeline:
            ms = _ms_since(self._t0) if self._t0 else 0
            await self._on_pipeline(stage, status, ms, extra)

    # ── Turn Lifecycle ──────────────────────────────────────────────

    async def start_turn(self, transcript: str, images=None) -> None:
        """Start a new agent turn."""
        if self._active:
            await self.cancel_turn()

        self._active = True
        self._t0 = time.monotonic()
        self._got_first_token = False
        self._got_first_audio = False

        muted = self._tts_muted_fn()

        # Begin tracing this turn
        self._turn = self._tracer.begin_turn(transcript)

        if not muted and self._tts_pool:
            self._tracer.begin(self._turn, "tts_pool")

            # Get TTS from pool (instant if warm, blocks if cold)
            self._tts = await self._tts_pool.get(
                on_audio=self._on_tts_audio,
                on_done=self._on_tts_done,
            )
            self._t_tts_conn = time.monotonic()
            self._tracer.end(self._turn, "tts_pool")

            # Create player
            self._player = self._player_cls(
                websocket=self._websocket,
                stream_sid=self._stream_sid,
                on_done=self._on_playback_done,
            )

        # Start LLM
        self._tracer.begin(self._turn, "llm")
        await self._pipeline("llm", "active")
        await self._llm.start(transcript, images=images)

        if not muted and self._tts_pool:
            tts_ms = int((self._t_tts_conn - self._t0) * 1000)
            log.info(f"Turn started  (TTS {tts_ms}ms = {tts_ms}ms setup)")
        elif not self._tts_pool:
            log.info("Turn started  (browser TTS)")
        else:
            log.info("Turn started  (TTS muted)")

    async def cancel_turn(self) -> None:
        """Cancel current turn, preserve history."""
        if not self._active:
            return

        elapsed = _ms_since(self._t0) if self._t0 else 0
        self._active = False

        # Mark turn as cancelled (ends all open spans)
        self._tracer.cancel_turn(self._turn)

        # Cancel in order: LLM -> TTS -> Player
        await self._llm.cancel()

        if self._tts:
            await self._tts.cancel()
            self._tts = None

        if self._player:
            await self._player.stop_and_clear()
            self._player = None

        await self._pipeline("llm", "idle")
        await self._pipeline("tts", "idle")
        await self._pipeline("playing", "idle")
        log.info(f"Turn cancelled at +{elapsed}ms (history preserved)")

    async def cleanup(self) -> None:
        """Final cleanup when call ends."""
        if self._active:
            await self.cancel_turn()

    # ── Internal Callbacks ──────────────────────────────────────────

    async def _on_tool_call_wrapper(self, name: str, args) -> None:
        """Intercept tool calls: clear TTS buffer so post-tool speech starts fresh."""
        # Discard buffered TTS text and queued audio for the pre-tool response
        if self._tts:
            self._tts._buffer = ""
            # Clear the synthesis queue
            while not self._tts._queue.empty():
                try:
                    self._tts._queue.get_nowait()
                except Exception:
                    break

        # Tell browser to stop playing pre-tool audio
        if self._on_pipeline:
            await self._pipeline("playing", "idle")

        # Forward to external handler (sends tool_call msg to browser)
        if self._external_on_tool_call:
            await self._external_on_tool_call(name, args)

    async def _on_llm_token(self, token: str) -> None:
        """LLM produced a token -> feed to TTS (unless muted)."""
        if not self._active:
            return

        if not self._got_first_token:
            self._got_first_token = True
            self._t_first_token = time.monotonic()
            self._tracer.mark(self._turn, "llm_first_token")
            llm_ms = _ms_since(self._t0)
            if self._tts:
                self._tracer.begin(self._turn, "tts")
                await self._pipeline("tts", "active", llm_ttft=llm_ms)
            else:
                await self._pipeline("llm", "active", llm_ttft=llm_ms)
            log.info(f"⏱  LLM first token  +{llm_ms}ms")

        if self._tts:
            # Strip markdown artifacts that TTS would read aloud
            clean = token.replace("*", "").replace("#", "").replace("`", "")
            if clean:
                await self._tts.send(clean)

        if self._on_response_token:
            await self._on_response_token(token)

    async def _on_llm_done(self) -> None:
        """LLM finished -> flush TTS or signal done directly."""
        if not self._active:
            return
        self._tracer.end(self._turn, "llm")
        llm_total = int((time.monotonic() - self._t_first_token) * 1000) if self._got_first_token else _ms_since(self._t0)
        await self._pipeline("llm", "done", llm_total=llm_total)

        if self._tts:
            await self._tts.flush()
        else:
            # TTS muted — turn is done once LLM finishes
            total = _ms_since(self._t0)
            log.info(f"⏱  Turn complete (muted)  +{total}ms total")
            self._active = False
            if self._on_pipeline:
                await self._pipeline("playing", "done", turn_total=total)
            self._on_done()

    async def _on_tts_audio(self, audio_base64: str) -> None:
        """TTS produced audio -> send to player."""
        if not self._active or not self._player:
            return

        if not self._got_first_audio:
            self._got_first_audio = True
            self._t_first_audio = time.monotonic()
            self._tracer.mark(self._turn, "tts_first_audio")
            self._tracer.begin(self._turn, "player")
            ttft = _ms_since(self._t0)
            tts_latency = int((self._t_first_audio - self._t_first_token) * 1000) if self._got_first_token else 0
            await self._pipeline("playing", "active", tts_latency=tts_latency, total_ttfa=ttft)
            log.info(f"⏱  TTS first audio  +{ttft}ms  (TTS latency {tts_latency}ms)")

        await self._player.send_chunk(audio_base64)

    async def _on_tts_done(self) -> None:
        """TTS finished -> tell player no more chunks coming."""
        if not self._active or not self._player:
            return
        self._tracer.end(self._turn, "tts")
        tts_total = int((time.monotonic() - self._t_first_token) * 1000) if self._got_first_token else _ms_since(self._t0)
        await self._pipeline("tts", "done", tts_total=tts_total)
        self._player.mark_tts_done()

    def _on_playback_done(self) -> None:
        """Player finished -> turn is complete."""
        if not self._active:
            return

        self._tracer.end(self._turn, "player")

        total = _ms_since(self._t0)
        log.info(f"⏱  Turn complete    +{total}ms total")

        self._active = False
        self._tts = None
        self._player = None

        if self._on_pipeline:
            asyncio.ensure_future(self._pipeline("playing", "done", turn_total=total))

        self._on_done()
