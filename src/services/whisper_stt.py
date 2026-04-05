"""
Server-side STT using Silero VAD + faster-whisper.

Server-side STT service. Callback interface:
on_start_of_turn, on_end_of_turn, on_interim.

Audio input: Int16 PCM at 16kHz (from browser AudioWorklet).
VAD: Silero v6 ONNX (bundled with faster-whisper).
Transcription: faster-whisper (CTranslate2 backend).
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Awaitable, Optional

import numpy as np

from ..log import ServiceLogger

log = ServiceLogger("WhisperSTT")

# Silero VAD frame size: 512 samples at 16kHz = 32ms
VAD_FRAME_SAMPLES = 512
VAD_CONTEXT_SAMPLES = 64  # context prepended to each frame
SAMPLE_RATE = 16000

# Max speech buffer: 30s to prevent unbounded growth
MAX_SPEECH_SAMPLES = SAMPLE_RATE * 30

# Interim transcription interval (seconds of speech)
INTERIM_INTERVAL_S = 2.0


class WhisperSTTService:
    """
    Server-side VAD + transcription service.

    Pipeline:
    1. send(audio_bytes) queues Int16 PCM chunks
    2. Processing loop runs Silero VAD frame-by-frame
    3. Speech start -> on_start_of_turn (barge-in)
    4. Silence >= silence_duration_ms -> transcribe -> on_end_of_turn(text)
    5. Periodic interim transcription -> on_interim(text)
    """

    def __init__(
        self,
        on_end_of_turn: Callable[[str], Awaitable[None]],
        on_start_of_turn: Callable[[], Awaitable[None]],
        on_interim: Optional[Callable[[str], Awaitable[None]]] = None,
        model_size: str = "base",
        language: str = "en",
        vad_threshold: float = 0.5,
        silence_duration_ms: int = 1500,
    ):
        self._on_end_of_turn = on_end_of_turn
        self._on_start_of_turn = on_start_of_turn
        self._on_interim = on_interim
        self._model_size = model_size
        self._language = language
        self._vad_threshold = vad_threshold
        self._silence_frames = int(silence_duration_ms / 32)  # 32ms per frame

        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self._speech_start_time = 0
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Loaded on start()
        self._whisper_model = None
        self._vad_session = None

    @property
    def is_active(self) -> bool:
        return self._running

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, lang: str) -> None:
        # Accept BCP-47 codes like "fr-FR" -> extract "fr" for Whisper
        self._language = lang.split("-")[0].lower() if lang else "en"

    async def start(self) -> None:
        """Load models and start processing loop."""
        if self._running:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_models)

        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
        log.connected()

    def _load_models(self) -> None:
        """Load Whisper and Silero VAD models (CPU, called in thread)."""
        from faster_whisper import WhisperModel

        log._logger.info(f"Loading Whisper model ({self._model_size})...")
        import torch
        use_gpu = torch.cuda.is_available()
        self._whisper_model = WhisperModel(
            self._model_size,
            device="cuda" if use_gpu else "cpu",
            compute_type="float16" if use_gpu else "int8",
        )
        log._logger.info(f"Whisper device: {'cuda' if use_gpu else 'cpu'}")

        self._load_silero_vad()
        log._logger.info(f"WhisperSTT loaded ({self._model_size} model)")

    def _load_silero_vad(self) -> None:
        """Load Silero VAD v6 ONNX model bundled with faster-whisper."""
        import onnxruntime as ort

        from faster_whisper.utils import get_assets_path
        vad_path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3

        self._vad_session = ort.InferenceSession(
            vad_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        # LSTM state for Silero v6
        self._vad_h = np.zeros((1, 1, 128), dtype=np.float32)
        self._vad_c = np.zeros((1, 1, 128), dtype=np.float32)
        # Context from previous frame (last 64 samples)
        self._vad_context = np.zeros(VAD_CONTEXT_SAMPLES, dtype=np.float32)

    def _run_vad(self, frame: np.ndarray) -> float:
        """Run Silero VAD v6 on a single 512-sample frame. Returns speech probability."""
        # Prepend context (64 samples) to frame (512 samples) -> input shape (1, 576)
        x = np.concatenate([self._vad_context, frame]).reshape(1, -1).astype(np.float32)
        self._vad_context = frame[-VAD_CONTEXT_SAMPLES:].copy()

        output, self._vad_h, self._vad_c = self._vad_session.run(
            None, {"input": x, "h": self._vad_h, "c": self._vad_c}
        )
        return output.item()

    def _reset_vad_state(self) -> None:
        """Reset VAD LSTM state between utterances."""
        self._vad_h = np.zeros((1, 1, 128), dtype=np.float32)
        self._vad_c = np.zeros((1, 1, 128), dtype=np.float32)
        self._vad_context = np.zeros(VAD_CONTEXT_SAMPLES, dtype=np.float32)

    async def send(self, audio_bytes: bytes) -> None:
        """Queue audio chunk for processing."""
        if not self._running:
            return
        await self._audio_queue.put(audio_bytes)

    async def stop(self) -> None:
        """Stop processing and clean up."""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None
        self._executor.shutdown(wait=False)
        log.disconnected()

    async def _process_loop(self) -> None:
        """Main processing loop: read audio, run VAD, transcribe."""
        pcm_buffer = np.array([], dtype=np.float32)  # unprocessed audio
        speech_buffer = np.array([], dtype=np.float32)  # speech segment
        in_speech = False
        silence_count = 0
        last_interim_len = 0  # speech samples at last interim

        try:
            while self._running:
                # Get audio with timeout to allow clean shutdown
                try:
                    audio_bytes = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # Convert Int16 PCM bytes to float32
                n_samples = len(audio_bytes) // 2
                if n_samples == 0:
                    continue
                int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                float32 = int16.astype(np.float32) / 32768.0
                pcm_buffer = np.concatenate([pcm_buffer, float32])

                # Process complete VAD frames
                while len(pcm_buffer) >= VAD_FRAME_SAMPLES:
                    frame = pcm_buffer[:VAD_FRAME_SAMPLES]
                    pcm_buffer = pcm_buffer[VAD_FRAME_SAMPLES:]

                    # Run VAD in thread pool (lightweight but still blocks)
                    loop = asyncio.get_event_loop()
                    prob = await loop.run_in_executor(
                        self._executor, self._run_vad, frame
                    )

                    if prob >= self._vad_threshold:
                        if not in_speech:
                            # Speech start
                            in_speech = True
                            silence_count = 0
                            last_interim_len = 0
                            self._speech_start_time = time.monotonic()
                            speech_buffer = np.array([], dtype=np.float32)
                            self._reset_vad_state()
                            await self._on_start_of_turn()

                        silence_count = 0
                        speech_buffer = np.concatenate([speech_buffer, frame])
                    else:
                        if in_speech:
                            speech_buffer = np.concatenate([speech_buffer, frame])
                            silence_count += 1

                            if silence_count >= self._silence_frames:
                                # End of speech — transcribe
                                await self._transcribe_and_emit(speech_buffer)
                                speech_buffer = np.array([], dtype=np.float32)
                                in_speech = False
                                silence_count = 0
                                last_interim_len = 0
                                self._reset_vad_state()

                    # Interim transcription
                    if (in_speech and self._on_interim
                            and len(speech_buffer) - last_interim_len
                            >= SAMPLE_RATE * INTERIM_INTERVAL_S):
                        last_interim_len = len(speech_buffer)
                        await self._transcribe_interim(speech_buffer)

                    # Force-transcribe at max length
                    if in_speech and len(speech_buffer) >= MAX_SPEECH_SAMPLES:
                        await self._transcribe_and_emit(speech_buffer)
                        # Keep trailing 0.5s for context continuity
                        keep = int(SAMPLE_RATE * 0.5)
                        speech_buffer = speech_buffer[-keep:]
                        last_interim_len = 0

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("Processing loop failed", e)

    async def _transcribe_and_emit(self, audio: np.ndarray) -> None:
        """Transcribe speech segment and emit end_of_turn."""
        if len(audio) < SAMPLE_RATE * 0.1:  # skip very short segments (<100ms)
            return

        t0 = time.monotonic()
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            self._executor, self._transcribe, audio
        )
        stt_ms = int((time.monotonic() - t0) * 1000)

        if transcript.strip():
            await self._on_end_of_turn(transcript.strip(), stt_ms=stt_ms)

    async def _transcribe_interim(self, audio: np.ndarray) -> None:
        """Transcribe current speech buffer for interim display."""
        if not self._on_interim or len(audio) < SAMPLE_RATE * 0.3:
            return

        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            self._executor, self._transcribe, audio
        )

        if transcript.strip():
            await self._on_interim(transcript.strip())

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run faster-whisper transcription (called in thread pool)."""
        try:
            segments, _ = self._whisper_model.transcribe(
                audio,
                language=self._language if self._language != "auto" else None,
                beam_size=1,
                word_timestamps=False,
                vad_filter=False,  # we already did VAD
            )
            return " ".join(seg.text for seg in segments)
        except Exception as e:
            log.error("Transcription failed", e)
            return ""
