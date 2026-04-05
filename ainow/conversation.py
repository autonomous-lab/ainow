"""
The main event loop for AINow.

This is the explicit, readable loop that drives the entire system:

    while connected:
        event = receive()                               # I/O (from queue)
        state, actions = process_event(state, event)    # PURE
        for action in actions:
            dispatch(action)                            # I/O

Events come from:
- Twilio WebSocket (audio packets)
- Deepgram Flux (turn events)
- Agent (playback complete)
"""

import os
import json
import uuid
import asyncio
from typing import Optional

from fastapi import WebSocket

from .types import (
    AppState,
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
    FluxStartOfTurnEvent, FluxEndOfTurnEvent, AgentTurnDoneEvent,
    FeedFluxAction, StartAgentTurnAction, ResetAgentTurnAction,
)
from .state import process_event
from .services.flux import FluxService
from .services.tts_pool import TTSPool
from .services.twilio_client import parse_twilio_message
from .services.browser_player import BrowserPlayer
from .agent import Agent
from .tracer import Tracer
from .log import Logger, get_logger

logger = get_logger("ainow.conversation")


async def run_conversation_over_twilio(websocket: WebSocket) -> None:
    """
    Main event loop for a single call.

    1. Create shared event queue
    2. Create Flux service (always-on STT + turn detection)
    3. Start Twilio reader
    4. On StreamStart, create Agent
    5. Process events through pure state machine
    6. Dispatch actions inline
    """
    event_log = Logger(verbose=False)
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    tracer = Tracer()

    agent: Optional[Agent] = None
    tts_pool = TTSPool(pool_size=1, ttl=8.0)
    stream_sid: Optional[str] = None

    # ── Flux Callbacks (push events to queue) ───────────────────────

    async def on_flux_end_of_turn(transcript: str) -> None:
        await event_queue.put(FluxEndOfTurnEvent(transcript=transcript))

    async def on_flux_start_of_turn() -> None:
        await event_queue.put(FluxStartOfTurnEvent())

    # ── Create Flux Service ─────────────────────────────────────────

    flux = FluxService(
        on_end_of_turn=on_flux_end_of_turn,
        on_start_of_turn=on_flux_start_of_turn,
    )

    # ── Twilio WebSocket Reader ─────────────────────────────────────

    async def read_twilio() -> None:
        """Background task to read from Twilio and push to event queue."""
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                event = parse_twilio_message(data)
                if event:
                    await event_queue.put(event)
                    if isinstance(event, StreamStopEvent):
                        break
        except Exception as e:
            event_log.error("Twilio reader", e)
            await event_queue.put(StreamStopEvent())

    # ── Initialize ──────────────────────────────────────────────────

    state = AppState()
    reader_task = asyncio.create_task(read_twilio())

    try:
        while True:
            # ─── RECEIVE ────────────────────────────────────────────
            event = await event_queue.get()

            event_log.event(event)

            # Initialize services on stream start
            if isinstance(event, StreamStartEvent):
                stream_sid = event.stream_sid
                await flux.start()
                await tts_pool.start()
                agent = Agent(
                    websocket=websocket,
                    stream_sid=event.stream_sid,
                    on_done=lambda: event_queue.put_nowait(AgentTurnDoneEvent()),
                    tts_pool=tts_pool,
                    tracer=tracer,
                )

            # ─── UPDATE (pure) ──────────────────────────────────────
            old_phase = state.phase
            state, actions = process_event(state, event)
            event_log.transition(old_phase, state.phase)

            # ─── DISPATCH (side effects) ────────────────────────────
            for action in actions:
                event_log.action(action)
                if isinstance(action, FeedFluxAction):
                    await flux.send(action.audio_bytes)

                elif isinstance(action, StartAgentTurnAction):
                    if agent:
                        await agent.start_turn(action.transcript)

                elif isinstance(action, ResetAgentTurnAction):
                    if agent:
                        await agent.cancel_turn()

            # ─── EXIT CHECK ─────────────────────────────────────────
            if isinstance(event, StreamStopEvent):
                break

    except Exception as e:
        event_log.error("Call loop", e)
        raise

    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass

        if agent:
            await agent.cleanup()

        await tts_pool.stop()
        await flux.stop()

        # Save trace
        call_id = stream_sid or "unknown"
        tracer.save(call_id)

        Logger.websocket_disconnected()


async def run_conversation_over_browser(websocket: WebSocket) -> None:
    """
    Main event loop for a browser voice session.

    Browser streams mic audio via AudioWorklet.
    Server handles STT (Whisper or Flux) + LLM + TTS and sends audio back.
    """
    event_log = Logger(verbose=False)
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    tracer = Tracer()

    agent: Optional[Agent] = None
    stream_sid = str(uuid.uuid4())

    # ── Shared state ──────────────────────────────────────────────
    custom_system_prompt: Optional[str] = None
    pending_confirms: dict = {}  # confirm_id -> asyncio.Future
    pending_browser_tools: dict = {}  # request_id -> asyncio.Future

    # ── Browser TTS (Chrome speechSynthesis) ───────────────────
    use_browser_tts = not bool(os.getenv("SERVER_TTS"))  # default: browser TTS

    # ── TTS Pool (Fish Speech, local Kokoro, or ElevenLabs) ────

    tts_pool = None
    use_fish_tts = bool(os.getenv("FISH_TTS_URL"))
    use_local_tts = bool(os.getenv("LOCAL_TTS_VOICE"))
    if not use_browser_tts:
        if use_fish_tts:
            from .services.fish_tts import FishTTSPool
            fish_url = os.getenv("FISH_TTS_URL", "http://localhost:8080")
            fish_ref = os.getenv("FISH_VOICE_ID", "") or None
            tts_pool = FishTTSPool(api_url=fish_url, reference_id=fish_ref)
        elif use_local_tts:
            from .services.local_tts import LocalTTSPool
            tts_pool = LocalTTSPool()
        else:
            tts_pool = TTSPool(pool_size=1, ttl=8.0, output_format="pcm_16000")

    # ── STT Backend ────────────────────────────────────────────────
    # Browser STT mode: browser handles speech recognition, no server STT needed.
    # Otherwise: Flux (if Deepgram key) or Whisper (local).

    use_browser_stt = bool(os.getenv("BROWSER_STT"))  # Browser speech recognition (no Whisper needed)
    use_flux = bool(os.getenv("DEEPGRAM_API_KEY"))
    stt = None  # FluxService or WhisperSTTService (None when browser STT)

    async def on_stt_end_of_turn(transcript: str, stt_ms: int = 0) -> None:
        # Echo final transcript to browser for chat display
        try:
            msg = {
                "type": "user_transcript",
                "transcript": transcript,
            }
            if stt_ms > 0:
                msg["stt_ms"] = stt_ms
            await websocket.send_text(json.dumps(msg))
        except Exception:
            pass
        await event_queue.put(FluxEndOfTurnEvent(transcript=transcript))

    async def on_stt_start_of_turn() -> None:
        await event_queue.put(FluxStartOfTurnEvent())

    async def on_stt_interim(transcript: str) -> None:
        try:
            await websocket.send_text(json.dumps({
                "type": "stt_interim",
                "transcript": transcript,
            }))
        except Exception:
            pass

    if not use_browser_stt:
        if use_flux:
            stt = FluxService(
                on_end_of_turn=on_stt_end_of_turn,
                on_start_of_turn=on_stt_start_of_turn,
                encoding="linear16",
                sample_rate=16000,
            )
        else:
            from .services.whisper_stt import WhisperSTTService
            stt = WhisperSTTService(
                on_end_of_turn=on_stt_end_of_turn,
                on_start_of_turn=on_stt_start_of_turn,
                on_interim=on_stt_interim,
                model_size=os.getenv("WHISPER_MODEL", "small"),
                language=os.getenv("WHISPER_LANG", "en"),
            )

    # ── TTS Mute State ─────────────────────────────────────────────
    tts_muted = False

    # ── Browser WebSocket Reader ──────────────────────────────────

    async def read_browser() -> None:
        """Background task to read from browser and push to event queue."""
        try:
            while True:
                message = await websocket.receive()
                msg_type = message.get("type", "")

                if msg_type == "websocket.receive":
                    if message.get("bytes") and stt:
                        # Binary frame = audio data (server STT)
                        await event_queue.put(MediaEvent(audio_bytes=message["bytes"]))
                    elif message.get("text"):
                        # Text frame = control message or transcript
                        data = json.loads(message["text"])
                        ctrl_type = data.get("type")

                        if ctrl_type == "end_of_turn":
                            # Browser STT finished — user done speaking
                            transcript = data.get("transcript", "").strip()
                            images = tuple(data.get("images", []))
                            if transcript or images:
                                await event_queue.put(FluxEndOfTurnEvent(transcript=transcript, images=images))
                        elif ctrl_type == "set_tts_mute":
                            nonlocal tts_muted
                            tts_muted = bool(data.get("muted", False))
                            logger.info(f"TTS mute set to: {tts_muted}")
                        elif ctrl_type == "start_of_turn":
                            # Browser detected user speaking (barge-in)
                            await event_queue.put(FluxStartOfTurnEvent())
                        elif ctrl_type == "set_lang":
                            lang = data.get("lang", "en-US")
                            if hasattr(tts_pool, 'lang'):
                                tts_pool.lang = lang
                            if hasattr(stt, 'language'):
                                stt.language = lang
                            logger.info(f"Language set to: {lang}")
                        elif ctrl_type == "set_system_prompt":
                            custom_system_prompt = data.get("prompt", "")
                            if agent:
                                agent.set_system_prompt(custom_system_prompt)
                            logger.info(f"System prompt updated ({len(custom_system_prompt)} chars)")
                        elif ctrl_type == "set_voice":
                            voice = data.get("voice", "")
                            if hasattr(tts_pool, 'voice'):
                                tts_pool.voice = voice or None
                                logger.info(f"Voice set to: {voice or 'default'}")
                        elif ctrl_type == "tool_confirm_response":
                            confirm_id = data.get("confirm_id", "")
                            approved = data.get("approved", False)
                            fut = pending_confirms.pop(confirm_id, None)
                            if fut and not fut.done():
                                fut.set_result(approved)
                        elif ctrl_type == "browser_tool_result":
                            request_id = data.get("request_id", "")
                            result = data.get("result", "")
                            fut = pending_browser_tools.pop(request_id, None)
                            if fut and not fut.done():
                                fut.set_result(result)
                        elif ctrl_type == "clear_session":
                            if agent:
                                agent.clear_history()
                            logger.info("Session cleared (history reset)")
                        elif ctrl_type == "restore_history":
                            if agent:
                                agent.clear_history()
                                messages = data.get("messages", [])
                                agent.restore_history(messages)
                            logger.info(f"History restored ({len(data.get('messages', []))} messages)")
                        elif ctrl_type == "stop":
                            await event_queue.put(StreamStopEvent())
                            break
                elif msg_type == "websocket.disconnect":
                    await event_queue.put(StreamStopEvent())
                    break
        except Exception as e:
            event_log.error("Browser reader", e)
            await event_queue.put(StreamStopEvent())

    # ── Initialize ────────────────────────────────────────────────

    state = AppState()

    # Immediately fire StreamStartEvent (no Twilio handshake needed)
    await event_queue.put(StreamStartEvent(stream_sid=stream_sid))

    reader_task = asyncio.create_task(read_browser())

    try:
        while True:
            # ─── RECEIVE ──────────────────────────────────────────
            event = await event_queue.get()

            event_log.event(event)

            # Initialize services on stream start
            if isinstance(event, StreamStartEvent):
                if stt:
                    await stt.start()
                if tts_pool:
                    await tts_pool.start()
                async def on_pipeline(stage: str, status: str, ms: int = 0, extra: dict = None) -> None:
                    try:
                        msg = {
                            "type": "pipeline",
                            "stage": stage,
                            "status": status,
                            "ms": ms,
                        }
                        if extra:
                            msg.update(extra)
                        await websocket.send_text(json.dumps(msg))
                    except Exception:
                        pass

                async def on_response_token(token: str) -> None:
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "transcript",
                            "token": token,
                        }))
                    except Exception:
                        pass

                async def on_tool_call(name: str, args) -> None:
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "tool_call",
                            "name": name,
                            "arguments": args,
                        }))
                    except Exception:
                        pass

                async def on_tool_result(tool_call_id: str, name: str, result: str) -> None:
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "tool_result",
                            "id": tool_call_id,
                            "name": name,
                            "result": result[:2000],
                        }))
                    except Exception:
                        pass

                async def on_tool_confirm(name: str, args) -> bool:
                    confirm_id = str(uuid.uuid4())
                    fut = asyncio.get_event_loop().create_future()
                    pending_confirms[confirm_id] = fut
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "tool_confirm",
                            "confirm_id": confirm_id,
                            "name": name,
                            "arguments": args,
                        }))
                        return await asyncio.wait_for(fut, timeout=60.0)
                    except (asyncio.TimeoutError, Exception):
                        return False
                    finally:
                        pending_confirms.pop(confirm_id, None)

                async def on_browser_tool(name: str, args: dict) -> str:
                    request_id = str(uuid.uuid4())
                    fut = asyncio.get_event_loop().create_future()
                    pending_browser_tools[request_id] = fut
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "browser_tool_call",
                            "request_id": request_id,
                            "name": name,
                            "arguments": args,
                        }))
                        result = await asyncio.wait_for(fut, timeout=30.0)
                        return result
                    except asyncio.TimeoutError:
                        return "Error: Browser tool timed out after 30s"
                    except Exception as e:
                        return f"Error: {e}"
                    finally:
                        pending_browser_tools.pop(request_id, None)

                agent = Agent(
                    websocket=websocket,
                    stream_sid=stream_sid,
                    on_done=lambda: event_queue.put_nowait(AgentTurnDoneEvent()),
                    tts_pool=tts_pool,
                    tracer=tracer,
                    player_cls=BrowserPlayer if tts_pool else None,
                    on_response_token=on_response_token,
                    on_pipeline=on_pipeline,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                    tts_muted_fn=lambda: tts_muted,
                    system_prompt=custom_system_prompt,
                    on_tool_confirm=on_tool_confirm,
                    on_browser_tool=on_browser_tool,
                )

                # Send config to browser (sample rate for audio playback)
                sample_rate = 24000 if (use_local_tts or use_fish_tts) else 16000
                from .services.llm import SYSTEM_PROMPT
                try:
                    await websocket.send_text(json.dumps({
                        "type": "config",
                        "sample_rate": sample_rate,
                        "use_browser_stt": use_browser_stt,
                        "use_browser_tts": use_browser_tts,
                        "system_prompt": custom_system_prompt or SYSTEM_PROMPT,
                    }))
                except Exception:
                    pass

            # ─── UPDATE (pure) ────────────────────────────────────
            old_phase = state.phase
            state, actions = process_event(state, event)
            event_log.transition(old_phase, state.phase)

            # Send phase changes to browser
            if old_phase != state.phase:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "state",
                        "phase": state.phase.name.lower(),
                    }))
                except Exception:
                    pass

            # ─── DISPATCH (side effects) ──────────────────────────
            for action in actions:
                event_log.action(action)
                if isinstance(action, FeedFluxAction):
                    if stt:
                        await stt.send(action.audio_bytes)

                elif isinstance(action, StartAgentTurnAction):
                    if agent:
                        await agent.start_turn(action.transcript, images=action.images or ())

                elif isinstance(action, ResetAgentTurnAction):
                    if agent:
                        await agent.cancel_turn()

            # ─── EXIT CHECK ───────────────────────────────────────
            if isinstance(event, StreamStopEvent):
                break

    except Exception as e:
        event_log.error("Browser loop", e)
        raise

    finally:
        # Resolve all pending tool confirms as denied
        for fut in pending_confirms.values():
            if not fut.done():
                fut.set_result(False)
        pending_confirms.clear()

        # Resolve all pending browser tools as error
        for fut in pending_browser_tools.values():
            if not fut.done():
                fut.set_result("Error: Connection closed")
        pending_browser_tools.clear()

        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass

        if agent:
            await agent.cleanup()

        if tts_pool:
            await tts_pool.stop()
        if stt:
            await stt.stop()

        # Save trace
        tracer.save(stream_sid)

        Logger.websocket_disconnected()
