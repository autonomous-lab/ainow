"""
Fish Speech TTS service (self-hosted).

Connects to a local Fish Speech API server (openaudio-s1-mini or similar).
Same interface as LocalTTSService for drop-in use.

Start the Fish Speech server:
    python -m tools.api_server \
        --listen 0.0.0.0:8080 \
        --llama-checkpoint-path checkpoints/openaudio-s1-mini \
        --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \
        --decoder-config-name modded_dac_vq

Env vars:
    FISH_TTS_URL      - Server URL (default: http://localhost:8080)
    FISH_VOICE_ID     - Reference voice ID (optional, for voice cloning)
"""

import re
import base64
import asyncio
import struct
from typing import Optional, Callable, Awaitable

import httpx
import numpy as np

from ..log import ServiceLogger

log = ServiceLogger("FishTTS")

# Sentence boundary: split on .!? followed by whitespace
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')
_MAX_WORDS = 30

# Target sample rate for browser playback
TARGET_SAMPLE_RATE = 24000

# PCM chunk size: ~100ms at 24kHz 16-bit mono = 4800 bytes
CHUNK_SIZE = 4800

_SENTINEL = None


class FishTTSService:
    """
    Fish Speech TTS via HTTP streaming.

    Buffers LLM tokens, splits on sentence boundaries, sends each
    sentence to Fish Speech /v1/tts with streaming=true, and forwards
    PCM audio chunks via callback.
    """

    def __init__(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
        api_url: str = "http://localhost:8080",
        reference_id: Optional[str] = None,
        lang: str = "en-US",
    ):
        self._on_audio = on_audio
        self._on_done = on_done
        self._api_url = api_url.rstrip("/")
        self._reference_id = reference_id
        self._lang = lang

        self._buffer = ""
        self._running = False
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None

    def bind(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ) -> None:
        self._on_audio = on_audio
        self._on_done = on_done

    @property
    def is_active(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True
        self._buffer = ""
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._synthesis_worker())
        log.connected()

    async def send(self, text: str) -> None:
        if not self._running:
            return
        self._buffer += text
        self._try_chunk()

    async def flush(self) -> None:
        if not self._running:
            return

        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            self._queue.put_nowait(remaining)

        self._queue.put_nowait(_SENTINEL)

        if self._worker:
            await self._worker
            self._worker = None

        await self._on_done()

    async def stop(self) -> None:
        await self.flush()
        self._running = False
        if self._client:
            await self._client.aclose()
            self._client = None
        log.disconnected()

    async def cancel(self) -> None:
        self._running = False
        self._buffer = ""
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        self._queue = None
        if self._client:
            await self._client.aclose()
            self._client = None
        log.cancelled()

    # ── Text chunking ────────────────────────────────────────────

    def _try_chunk(self) -> None:
        # Split on sentence boundary (.!?)
        parts = _SENTENCE_RE.split(self._buffer, maxsplit=1)
        if len(parts) > 1:
            chunk = parts[0].strip()
            self._buffer = parts[1]
            if chunk:
                self._queue.put_nowait(chunk)
            return

        # Word-count fallback
        words = self._buffer.split()
        if len(words) >= _MAX_WORDS:
            chunk = " ".join(words)
            self._buffer = ""
            if chunk.strip():
                self._queue.put_nowait(chunk.strip())

    # ── Synthesis worker ─────────────────────────────────────────

    async def _synthesis_worker(self) -> None:
        while self._running:
            text = await self._queue.get()
            if text is _SENTINEL:
                break
            await self._synthesize(text)

    async def _synthesize(self, text: str) -> None:
        if not text.strip() or not self._running or not self._client:
            return

        try:
            body = {
                "text": text,
                "format": "wav",
                "streaming": True,
                "normalize": True,
            }
            if self._reference_id:
                body["reference_id"] = self._reference_id

            async with self._client.stream(
                "POST", f"{self._api_url}/v1/tts", json=body
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    log.error(f"Fish TTS HTTP {resp.status_code}: {error_body[:200]}")
                    return

                await self._stream_wav_response(resp)

        except httpx.ConnectError:
            log.error(f"Cannot connect to Fish TTS at {self._api_url}")
        except Exception as e:
            log.error(f"Synthesis failed for: {text[:50]}...", e)

    async def _stream_wav_response(self, resp) -> None:
        """Parse streaming WAV response and emit PCM chunks."""
        header_buf = b""
        header_parsed = False
        sample_rate = 44100
        pcm_buf = b""

        async for raw_chunk in resp.aiter_bytes(1024):
            if not self._running:
                break

            if not header_parsed:
                header_buf += raw_chunk
                # Parse WAV header once we have enough data
                result = _parse_wav_header(header_buf)
                if result is not None:
                    sample_rate, data_offset = result
                    pcm_buf = header_buf[data_offset:]
                    header_parsed = True
                continue

            pcm_buf += raw_chunk

            # Emit complete chunks
            pcm_buf = await self._emit_chunks(pcm_buf, sample_rate)

        # Emit remaining data
        if pcm_buf and self._running:
            await self._emit_chunks(pcm_buf, sample_rate, flush=True)

    async def _emit_chunks(
        self, pcm_buf: bytes, sample_rate: int, flush: bool = False
    ) -> bytes:
        """Send complete CHUNK_SIZE pieces from pcm_buf. Returns remainder."""
        target_chunk = CHUNK_SIZE
        if sample_rate != TARGET_SAMPLE_RATE:
            # Adjust read size to produce ~CHUNK_SIZE after resampling
            ratio = sample_rate / TARGET_SAMPLE_RATE
            target_chunk = int(CHUNK_SIZE * ratio)
            # Ensure even number of bytes (int16)
            target_chunk = target_chunk - (target_chunk % 2)

        while len(pcm_buf) >= target_chunk:
            if not self._running:
                return b""
            chunk = pcm_buf[:target_chunk]
            pcm_buf = pcm_buf[target_chunk:]
            if sample_rate != TARGET_SAMPLE_RATE:
                chunk = _resample(chunk, sample_rate, TARGET_SAMPLE_RATE)
            audio_b64 = base64.b64encode(chunk).decode()
            await self._on_audio(audio_b64)

        # Flush remaining
        if flush and pcm_buf and self._running:
            if sample_rate != TARGET_SAMPLE_RATE:
                pcm_buf = _resample(pcm_buf, sample_rate, TARGET_SAMPLE_RATE)
            if pcm_buf:
                audio_b64 = base64.b64encode(pcm_buf).decode()
                await self._on_audio(audio_b64)
            return b""

        return pcm_buf


def _parse_wav_header(buf: bytes):
    """
    Parse WAV header, return (sample_rate, data_offset) or None if incomplete.

    Handles WAV files with extra chunks between "fmt " and "data".
    """
    if len(buf) < 12:
        return None

    # Verify RIFF/WAVE
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        return None

    pos = 12
    sample_rate = 44100  # fallback

    while pos + 8 <= len(buf):
        chunk_id = buf[pos:pos + 4]
        if pos + 8 > len(buf):
            return None
        chunk_size = struct.unpack_from("<I", buf, pos + 4)[0]

        if chunk_id == b"fmt ":
            if pos + 8 + 8 > len(buf):
                return None
            sample_rate = struct.unpack_from("<I", buf, pos + 12)[0]

        elif chunk_id == b"data":
            data_offset = pos + 8
            return (sample_rate, data_offset)

        pos += 8 + chunk_size
        # Chunks are word-aligned
        if chunk_size % 2:
            pos += 1

    return None  # Haven't found "data" chunk yet


def _resample(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linear interpolation resampling of PCM int16 data."""
    if not pcm_bytes or from_rate == to_rate:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(samples) == 0:
        return pcm_bytes

    ratio = to_rate / from_rate
    new_len = max(1, int(len(samples) * ratio))
    indices = np.linspace(0, len(samples) - 1, new_len)
    resampled = np.interp(indices, np.arange(len(samples)), samples.astype(np.float64))
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


class FishTTSPool:
    """
    Pool for FishTTSService instances.

    Same interface as LocalTTSPool / TTSPool so Agent doesn't need
    to know the difference.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8080",
        reference_id: Optional[str] = None,
        lang: str = "en-US",
        **kwargs,
    ):
        self._api_url = api_url
        self._reference_id = reference_id
        self._lang = lang

    @property
    def lang(self) -> str:
        return self._lang

    @lang.setter
    def lang(self, value: str) -> None:
        self._lang = value

    async def start(self) -> None:
        """Verify Fish Speech server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._api_url}/v1/health")
                if resp.status_code == 200:
                    log.info(f"Fish TTS server OK at {self._api_url}")
                else:
                    log.info(f"Fish TTS health check returned {resp.status_code}")
        except httpx.ConnectError:
            log.info(f"Fish TTS server not reachable at {self._api_url} (will retry on use)")
        except Exception as e:
            log.info(f"Fish TTS health check failed: {e}")

    async def stop(self) -> None:
        pass

    async def get(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ) -> FishTTSService:
        tts = FishTTSService(
            on_audio=on_audio,
            on_done=on_done,
            api_url=self._api_url,
            reference_id=self._reference_id,
            lang=self._lang,
        )
        await tts.start()
        return tts
