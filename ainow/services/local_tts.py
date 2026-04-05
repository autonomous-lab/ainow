"""
Local TTS service using Kokoro (hexgrad/Kokoro-82M).

Aggressive clause-level chunking for low time-to-first-audio:
- First chunk fires after ~4 words (or any punctuation)
- Subsequent chunks fire on clause boundaries or ~8 words
- Synthesis runs in thread pool, pipelined with LLM streaming

Kokoro supports native voices for: EN, FR, ES, IT, PT, ZH, HI
"""

import re
import base64
import asyncio
import numpy as np
from typing import Optional, Callable, Awaitable

from ..log import ServiceLogger

log = ServiceLogger("LocalTTS")

# Sentence boundary: only split on sentence-ending punctuation (.!?)
# Kokoro is fast enough locally — full sentences sound much more natural
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

# Word count fallback for very long sentences without a period
_MAX_WORDS = 30

# Language → (kokoro lang_code, default voice)
LANG_VOICES = {
    "en": ("a", "af_heart"),      # American English
    "fr": ("f", "ff_siwis"),      # French
    "es": ("e", "ef_dora"),       # Spanish
    "it": ("i", "if_sara"),       # Italian
    "pt": ("p", "pf_dora"),       # Portuguese
    "zh": ("z", "zf_xiaobei"),    # Chinese
    "hi": ("h", "hf_alpha"),      # Hindi
}

DEFAULT_LANG = ("a", "af_heart")

# Shared Kokoro model + per-language pipelines (loaded once)
_kokoro_model = None
_pipelines: dict = {}  # lang_code → KPipeline


def _load_model():
    """Load Kokoro model (once). Pipelines are created on demand."""
    global _kokoro_model
    if _kokoro_model is not None:
        return

    from kokoro import KModel

    log.info("Loading Kokoro model...")
    _kokoro_model = KModel(repo_id="hexgrad/Kokoro-82M")
    log.info("Kokoro model ready (24kHz)")

    # Pre-load English pipeline
    _get_pipeline("a")


def _get_pipeline(lang_code: str):
    """Get or create a KPipeline for a language (cached)."""
    if lang_code not in _pipelines:
        from kokoro import KPipeline
        log.info(f"Creating pipeline for lang_code={lang_code}")
        _pipelines[lang_code] = KPipeline(lang_code=lang_code, model=_kokoro_model)
    return _pipelines[lang_code]


def _resolve_lang(lang_str: str) -> tuple:
    """Map a language code (e.g. 'fr-FR' or 'fr') to (kokoro_lang_code, voice)."""
    short = lang_str.split("-")[0].lower()
    return LANG_VOICES.get(short, DEFAULT_LANG)


_SENTINEL = None  # marks end of chunk queue


class LocalTTSService:
    """
    Local TTS using Kokoro.

    Aggressive clause-level chunking: fires TTS on commas, short
    word counts, and sentence boundaries so audio starts playing
    while the LLM is still generating.

    Chunks are synthesized **sequentially** via an asyncio.Queue
    to guarantee audio order.
    """

    def __init__(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
        output_format: str = "pcm_24000",  # kept for interface compat
        lang: str = "en-US",
        voice: Optional[str] = None,
    ):
        self._on_audio = on_audio
        self._on_done = on_done
        self._lang = lang
        self._lang_code, default_voice = _resolve_lang(lang)
        self._voice = voice or default_voice

        self._buffer = ""
        self._running = False
        self._sent_first = False
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None

    def bind(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ) -> None:
        """Rebind callbacks (pool compat)."""
        self._on_audio = on_audio
        self._on_done = on_done

    @property
    def is_active(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Mark as ready and start the sequential synthesis worker."""
        self._running = True
        self._buffer = ""
        self._sent_first = False
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._synthesis_worker())
        log.connected()

    async def send(self, text: str) -> None:
        """Buffer text and fire TTS on clause boundaries or word count."""
        if not self._running:
            return

        self._buffer += text
        self._try_chunk()

    async def flush(self) -> None:
        """Synthesize any remaining buffered text, then signal done."""
        if not self._running:
            return

        # Push remaining buffer
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            self._queue.put_nowait(remaining)

        # Signal worker to finish
        self._queue.put_nowait(_SENTINEL)

        # Wait for worker to drain the queue
        if self._worker:
            await self._worker
            self._worker = None

        await self._on_done()

    async def stop(self) -> None:
        """Clean shutdown."""
        await self.flush()
        self._running = False
        log.disconnected()

    async def cancel(self) -> None:
        """Abort immediately."""
        self._running = False
        self._buffer = ""
        self._sent_first = False
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        self._queue = None
        log.cancelled()

    def _try_chunk(self) -> None:
        """Try to extract a complete sentence from the buffer and enqueue it."""
        # Strategy 1: Split on sentence boundary (.!?)
        parts = _SENTENCE_RE.split(self._buffer, maxsplit=1)
        if len(parts) > 1:
            chunk = parts[0].strip()
            self._buffer = parts[1]
            if chunk:
                self._sent_first = True
                self._queue.put_nowait(chunk)
            return

        # Strategy 2: Word-count fallback for very long sentences
        words = self._buffer.split()
        if len(words) >= _MAX_WORDS:
            chunk = " ".join(words)
            self._buffer = ""
            if chunk.strip():
                self._sent_first = True
                self._queue.put_nowait(chunk.strip())

    async def _synthesis_worker(self) -> None:
        """Sequential worker: pulls chunks from queue and synthesizes in order."""
        while self._running:
            text = await self._queue.get()
            if text is _SENTINEL:
                break
            await self._synthesize(text)

    async def _synthesize(self, text: str) -> None:
        """Generate audio with Kokoro and send via callback."""
        if not text.strip() or not self._running:
            return

        try:
            pcm_bytes = await asyncio.to_thread(
                self._generate_pcm, text, self._lang_code, self._voice
            )

            if not pcm_bytes or not self._running:
                return

            # Send in chunks (~100ms at 24kHz, 16-bit mono = 4800 bytes)
            chunk_size = 4800
            for i in range(0, len(pcm_bytes), chunk_size):
                if not self._running:
                    break
                chunk = pcm_bytes[i:i + chunk_size]
                audio_b64 = base64.b64encode(chunk).decode()
                await self._on_audio(audio_b64)

        except Exception as e:
            log.error(f"Synthesis failed for: {text[:50]}...", e)

    @staticmethod
    def _generate_pcm(text: str, lang_code: str, voice: str) -> bytes:
        """Generate PCM int16 bytes from text (runs in thread)."""
        global _kokoro_model
        if _kokoro_model is None:
            return b""

        pipeline = _get_pipeline(lang_code)

        # Kokoro's pipeline yields chunks; concatenate all
        audio_parts = []
        for result in pipeline(text, voice=voice):
            if result.audio is not None:
                audio_parts.append(result.audio.numpy())

        if not audio_parts:
            return b""

        audio_np = np.concatenate(audio_parts)
        audio_np = np.clip(audio_np, -1.0, 1.0)
        int16_data = (audio_np * 32767).astype(np.int16)
        return int16_data.tobytes()


class LocalTTSPool:
    """
    Lightweight pool for LocalTTSService.

    Loads the Kokoro model once on start().
    Same interface as TTSPool so Agent doesn't need to know the difference.
    """

    def __init__(self, lang: str = "en-US", **kwargs):
        self._lang = lang
        self._voice: Optional[str] = None

    @property
    def lang(self) -> str:
        return self._lang

    @lang.setter
    def lang(self, value: str) -> None:
        self._lang = value

    @property
    def voice(self) -> Optional[str]:
        return self._voice

    @voice.setter
    def voice(self, value: Optional[str]) -> None:
        self._voice = value

    async def start(self) -> None:
        """Load the TTS model (once)."""
        await asyncio.to_thread(_load_model)

    async def stop(self) -> None:
        """No-op."""
        pass

    async def get(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ) -> LocalTTSService:
        """Create a fresh LocalTTSService instance."""
        tts = LocalTTSService(on_audio=on_audio, on_done=on_done, lang=self._lang, voice=self._voice)
        await tts.start()
        return tts
