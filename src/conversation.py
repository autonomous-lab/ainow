"""
The main event loop for AINow.

This is the explicit, readable loop that drives the entire system:

    while connected:
        event = receive()                               # I/O (from queue)
        state, actions = process_event(state, event)    # PURE
        for action in actions:
            dispatch(action)                            # I/O

Events come from:
- Browser WebSocket (audio packets, transcripts)
- STT service (turn events)
- Agent (playback complete)
"""

import os
import json
import uuid
import asyncio
import re
from typing import Optional

from fastapi import WebSocket

from .types import (
    AppState, Phase,
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
    StartOfTurnEvent, EndOfTurnEvent, AgentTurnDoneEvent,
    FeedSTTAction, StartAgentTurnAction, ResetAgentTurnAction,
)
from .state import process_event
from .services.browser_player import BrowserPlayer
from .agent import Agent
from .tracer import Tracer
from .log import Logger, get_logger

logger = get_logger("ainow.conversation")


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_session_id(value: str, default: str) -> str:
    if not isinstance(value, str):
        return default
    sid = value.strip()
    if not sid:
        return default
    if not _SESSION_ID_RE.fullmatch(sid):
        return default
    return sid


def _get_str(payload: dict, key: str, default: str = "", max_len: int = 8192) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        return default
    value = value.strip()
    if not value:
        return default
    return value[:max_len]


def _get_bool(payload: dict, key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _get_image_list(payload: dict, key: str = "images", max_items: int = 8) -> list:
    """Extract an images list from a client message. Each item must be a
    {data, mime} dict where data is a base64 data URL or raw base64 string.
    Filters bad entries but keeps the dict shape.
    """
    value = payload.get(key, [])
    if not isinstance(value, list):
        return []
    out: list = []
    for item in value[:max_items]:
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        mime = item.get("mime")
        if not isinstance(data, str) or not data:
            continue
        if mime is not None and not isinstance(mime, str):
            continue
        out.append({"data": data, "mime": mime or ""})
    return out


async def run_conversation_over_browser(websocket: WebSocket) -> None:
    """
    Main event loop for a browser voice session.

    Browser streams mic audio via AudioWorklet.
    Server handles STT (Whisper) + LLM + TTS and sends audio back.
    """
    event_log = Logger(verbose=False)
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    tracer = Tracer()

    agent: Optional[Agent] = None
    stream_sid = str(uuid.uuid4())

    # ── Shared state ──────────────────────────────────────────────
    session_id = stream_sid  # Can be overridden by client
    from .services import agents as agent_store
    from .services.mcp import mcp_manager
    from .services.live_conversations import live_conversations
    agent_store.ensure_default()
    active_agent_name: str = agent_store.get_active()
    pending_confirms: dict = {}  # confirm_id -> (tool_name, asyncio.Future)
    auto_approved_tools: set = set()  # tool names whitelisted for this session

    async def _push_mcp_status() -> None:
        """Send the current MCP server status to the browser."""
        try:
            await websocket.send_text(json.dumps({
                "type": "mcp_status",
                "agent": active_agent_name,
                "servers": mcp_manager.loaded_servers(),
            }))
        except Exception:
            pass

    async def _activate_mcp_for(name: str, force: bool = False) -> None:
        """(Re)load MCP servers for an agent and notify the browser."""
        try:
            await mcp_manager.activate_agent(name, force=force)
        except Exception as e:
            logger.error(f"MCP activation failed for '{name}': {e}")
        await _push_mcp_status()
    pending_browser_tools: dict = {}  # request_id -> asyncio.Future

    # ── Browser TTS (Chrome speechSynthesis) ───────────────────
    use_browser_tts = not bool(os.getenv("SERVER_TTS"))  # default: browser TTS

    # ── TTS Pool (local Kokoro) ──────────────────────────────
    tts_pool = None
    use_local_tts = bool(os.getenv("LOCAL_TTS_VOICE"))
    if not use_browser_tts and use_local_tts:
        from .services.local_tts import LocalTTSPool
        tts_pool = LocalTTSPool()

    # ── STT Backend ────────────────────────────────────────────────
    # Browser STT mode: browser handles speech recognition, no server STT needed.
    # Otherwise: Whisper (local).

    use_browser_stt = bool(os.getenv("BROWSER_STT"))  # Browser speech recognition (no Whisper needed)
    stt = None  # WhisperSTTService (None when browser STT)

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
        await event_queue.put(EndOfTurnEvent(transcript=transcript))

    async def on_stt_start_of_turn() -> None:
        await event_queue.put(StartOfTurnEvent())

    async def on_stt_interim(transcript: str) -> None:
        try:
            await websocket.send_text(json.dumps({
                "type": "stt_interim",
                "transcript": transcript,
            }))
        except Exception:
            pass

    if not use_browser_stt:
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
        nonlocal active_agent_name
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
                        raw = message["text"]
                        if not isinstance(raw, str):
                            logger.warning("Ignoring non-string websocket text frame")
                            continue
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Ignoring invalid websocket JSON: {e}")
                            continue
                        if not isinstance(data, dict):
                            logger.warning("Ignoring websocket text frame with non-object payload")
                            continue
                        ctrl_type = data.get("type")
                        if not isinstance(ctrl_type, str):
                            logger.warning("Ignoring websocket control with non-string type")
                            continue
                        ctrl_type = ctrl_type.strip()
                        if not ctrl_type:
                            continue

                        allowed_ctrl = {
                            "end_of_turn", "set_tts_mute", "start_of_turn", "set_lang", "reload_mcp",
                            "set_agent", "switch_model", "set_voice", "tool_confirm_response",
                            "browser_tool_result", "set_session_id", "clear_session", "restore_history",
                            "save_session", "load_session", "list_sessions", "stop"
                        }
                        if ctrl_type not in allowed_ctrl:
                            logger.warning(f"Ignoring unknown control type: {ctrl_type}")
                            continue

                        if ctrl_type == "end_of_turn":
                            # Browser STT finished — user done speaking
                            transcript = _get_str(data, "transcript")
                            images = tuple(_get_image_list(data, "images"))
                            audio = data.get("audio")
                            if not isinstance(audio, str):
                                audio = None
                            if transcript or images or audio:
                                await event_queue.put(EndOfTurnEvent(transcript=transcript, images=images, audio=audio))
                        elif ctrl_type == "set_tts_mute":
                            nonlocal tts_muted
                            tts_muted = _get_bool(data, "muted", False)
                            logger.info(f"TTS mute set to: {tts_muted}")
                        elif ctrl_type == "start_of_turn":
                            # Browser detected user speaking (barge-in)
                            await event_queue.put(StartOfTurnEvent())
                        elif ctrl_type == "set_lang":
                            lang = _get_str(data, "lang", "en-US", 32)
                            if hasattr(tts_pool, 'lang'):
                                tts_pool.lang = lang
                            if hasattr(stt, 'language'):
                                stt.language = lang
                            # Persist as a per-agent preference
                            try:
                                agent_store.update_preferences(active_agent_name, {"lang": lang})
                            except Exception as e:
                                logger.warning(f"Failed to persist lang preference: {e}")
                            logger.info(f"Language set to: {lang}")
                        elif ctrl_type == "reload_mcp":
                            # Force-reload MCP servers for the current agent (after UI edit).
                            # Run in background so the WS reader doesn't block.
                            asyncio.create_task(_activate_mcp_for(active_agent_name, force=True))
                        elif ctrl_type == "set_agent":
                            new_name = _get_str(data, "name", "", 128)
                            if new_name and agent_store.exists(new_name):
                                active_agent_name = new_name
                                agent_store.set_active(new_name)
                                if agent:
                                    agent.set_agent(new_name)
                                logger.info(f"Agent switched to: {new_name}")
                                # Update live-conversations registry so scheduler targets the new agent
                                live_conversations.update_agent(stream_sid, new_name)
                                # Reload MCP servers for the new agent (background)
                                asyncio.create_task(_activate_mcp_for(new_name))
                                # Apply per-agent preferences (lang/voice) on the server side
                                prefs = agent_store.read_preferences(new_name)
                                pref_lang = prefs.get("lang")
                                pref_voice = prefs.get("voice")
                                if pref_lang:
                                    if hasattr(tts_pool, 'lang'):
                                        tts_pool.lang = pref_lang
                                    if hasattr(stt, 'language'):
                                        stt.language = pref_lang
                                if pref_voice and hasattr(tts_pool, 'voice'):
                                    tts_pool.voice = pref_voice or None
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "agent_switched",
                                        "name": new_name,
                                        "preferences": prefs,
                                    }))
                                except Exception as e:
                                    logger.warning(f"Failed to apply preference after agent switch: {e}")
                            else:
                                logger.warning(f"Cannot switch to unknown agent: {new_name}")
                        elif ctrl_type == "switch_model":
                            # Read from env vars (set by POST /api/models/{alias})
                            base_url = os.getenv("LLM_BASE_URL", "")
                            api_key = os.getenv("LLM_API_KEY", "not-needed")
                            model = os.getenv("LLM_MODEL", "")
                            if agent and base_url and model:
                                agent.switch_model(base_url, api_key, model)
                                logger.info(f"Model switched to: {model} @ {base_url}")
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "model_switched",
                                        "model": model,
                                    }))
                                except Exception:
                                    pass
                        elif ctrl_type == "set_voice":
                            voice = _get_str(data, "voice", "")
                            if hasattr(tts_pool, 'voice'):
                                tts_pool.voice = voice or None
                                logger.info(f"Voice set to: {voice or 'default'}")
                            # Persist as a per-agent preference
                            try:
                                agent_store.update_preferences(active_agent_name, {"voice": voice})
                            except Exception as e:
                                logger.warning(f"Failed to persist voice preference: {e}")
                        elif ctrl_type == "tool_confirm_response":
                            confirm_id = _get_str(data, "confirm_id", "")
                            approved = _get_bool(data, "approved", False)
                            always = _get_bool(data, "always", False)
                            entry = pending_confirms.pop(confirm_id, None)
                            if entry:
                                tool_name, fut = entry
                                if always and approved and tool_name:
                                    auto_approved_tools.add(tool_name)
                                    logger.info(f"Tool '{tool_name}' added to session auto-approve list")
                                if fut and not fut.done():
                                    fut.set_result(approved)
                        elif ctrl_type == "browser_tool_result":
                            request_id = _get_str(data, "request_id", "")
                            result = data.get("result", "")
                            fut = pending_browser_tools.pop(request_id, None)
                            if fut and not fut.done():
                                fut.set_result(result)
                        elif ctrl_type == "set_session_id":
                            nonlocal session_id
                            candidate = _get_str(data, "session_id", "", max_len=128)
                            session_id = _safe_session_id(candidate, session_id)
                            logger.info(f"Session ID set to: {session_id}")
                        elif ctrl_type == "clear_session":
                            if agent:
                                agent.clear_history()
                                # Refresh the CTX badge in the UI
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "pipeline",
                                        "stage": "llm",
                                        "status": "idle",
                                        "ms": 0,
                                        "context_used": agent._llm.context_used,
                                        "context_max": agent._llm.context_max,
                                    }))
                                except Exception:
                                    pass
                            logger.info("Session cleared (history reset)")
                        elif ctrl_type == "restore_history":
                            restored_count = 0
                            if agent:
                                agent.clear_history()
                                messages = data.get("messages", [])
                                if not isinstance(messages, list):
                                    messages = []
                                agent.restore_history(messages)
                                restored_count = len(agent.history)
                                # Refresh the CTX badge after restoring
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "pipeline",
                                        "stage": "llm",
                                        "status": "idle",
                                        "ms": 0,
                                        "context_used": agent._llm.context_used,
                                        "context_max": agent._llm.context_max,
                                    }))
                                except Exception:
                                    pass
                            logger.info(f"History restored ({restored_count} messages)")
                        elif ctrl_type == "save_session":
                            session_id = _get_str(data, "session_id", str(uuid.uuid4()), 128)
                            if agent:
                                session_id = _safe_session_id(session_id, str(uuid.uuid4()))
                                path = agent.save_session(session_id)
                                logger.info(f"Session saved: {session_id}")
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "session_saved",
                                        "session_id": session_id,
                                    }))
                                except Exception:
                                    pass
                        elif ctrl_type == "load_session":
                            sid = _get_str(data, "session_id", "", 128)
                            sid = _safe_session_id(sid, "")
                            if agent and sid and agent.load_session(sid):
                                # Send loaded messages to browser for display
                                history = agent.history
                                chat_msgs = []
                                for msg in history:
                                    role = msg.get("role", "")
                                    content = msg.get("content", "")
                                    if isinstance(content, list):
                                        content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
                                    if role in ("user", "assistant") and content:
                                        chat_msgs.append({"role": role, "text": content})
                                # Refresh the CTX badge after the load
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "pipeline",
                                        "stage": "llm",
                                        "status": "idle",
                                        "ms": 0,
                                        "context_used": agent._llm.context_used,
                                        "context_max": agent._llm.context_max,
                                    }))
                                except Exception:
                                    pass
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "session_loaded",
                                        "session_id": sid,
                                        "messages": chat_msgs,
                                    }))
                                except Exception as e:
                                    logger.warning(f"Failed to send session list: {e}")
                                logger.info(f"Session loaded: {sid} ({len(chat_msgs)} messages)")
                            else:
                                logger.warning(f"Session not found: {sid}")
                        elif ctrl_type == "list_sessions":
                            from .services.llm import LLMService
                            sessions = LLMService.list_sessions()
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "sessions_list",
                                    "sessions": sessions,
                                }))
                            except Exception:
                                pass
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

    # Register this conversation for the scheduler so it can find live
    # sessions to inject scheduled-task prompts into.
    async def _push_user_prompt(text: str) -> None:
        """Inject a user prompt as if the human typed it (used by scheduler)."""
        await event_queue.put(EndOfTurnEvent(transcript=text))
    live_conversations.register(
        stream_sid,
        agent_name=active_agent_name,
        websocket=websocket,
        event_queue=event_queue,
        push_user_prompt=_push_user_prompt,
    )

    # Immediately fire StreamStartEvent
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
                            "result": result[:32000],
                        }))
                    except Exception:
                        pass

                async def on_tool_confirm(name: str, args) -> bool:
                    # Session-level whitelist: skip the dialog entirely once
                    # the user has clicked "Always" for this tool name.
                    if name in auto_approved_tools:
                        return True
                    confirm_id = str(uuid.uuid4())
                    fut = asyncio.get_event_loop().create_future()
                    pending_confirms[confirm_id] = (name, fut)
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

                async def on_thinking(text: str, duration_s: float, done: bool = False) -> None:
                    """Stream thinking tokens or signal thinking-done."""
                    try:
                        if done:
                            await websocket.send_text(json.dumps({
                                "type": "thinking_done",
                                "duration": round(duration_s, 1),
                            }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "thinking_delta",
                                "token": text,
                            }))
                    except Exception:
                        pass

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
                    agent_name=active_agent_name,
                    on_tool_confirm=on_tool_confirm,
                    on_browser_tool=on_browser_tool,
                    on_thinking=on_thinking,
                )

                # Activate MCP servers for the current agent in the background.
                # npx-based servers can take 5–15s to spawn on Windows; we don't
                # want to block WebSocket setup. Tools will appear as soon as
                # they're ready (LLMService re-fetches schemas on every turn).
                asyncio.create_task(_activate_mcp_for(active_agent_name))

                # Send config to browser (sample rate for audio playback)
                sample_rate = 24000 if use_local_tts else 16000
                use_audio_llm = bool(os.getenv("AUDIO_LLM"))
                # Apply persisted per-agent preferences server-side too
                prefs = agent_store.read_preferences(active_agent_name)
                if prefs.get("lang"):
                    if hasattr(tts_pool, 'lang'):
                        tts_pool.lang = prefs["lang"]
                    if hasattr(stt, 'language'):
                        stt.language = prefs["lang"]
                if prefs.get("voice") and hasattr(tts_pool, 'voice'):
                    tts_pool.voice = prefs["voice"]
                try:
                    await websocket.send_text(json.dumps({
                        "type": "config",
                        "sample_rate": sample_rate,
                        "use_browser_stt": use_browser_stt,
                        "use_browser_tts": use_browser_tts,
                        "use_audio_llm": use_audio_llm,
                        "active_agent": active_agent_name,
                        "preferences": prefs,
                    }))
                except Exception:
                    pass

            # ─── UPDATE (pure) ────────────────────────────────────
            old_phase = state.phase
            state, actions = process_event(state, event)
            event_log.transition(old_phase, state.phase)

            # Send phase changes to browser
            if old_phase != state.phase:
                # Auto-save session when agent finishes a turn
                if old_phase == Phase.RESPONDING and state.phase == Phase.LISTENING and agent:
                    try:
                        path = agent.save_session(session_id)
                        logger.info(f"Session auto-saved: {session_id}")
                    except Exception as e:
                        logger.error(f"Session save failed: {e}")
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
                if isinstance(action, FeedSTTAction):
                    if stt:
                        await stt.send(action.audio_bytes)

                elif isinstance(action, StartAgentTurnAction):
                    if agent:
                        await agent.start_turn(action.transcript, images=action.images or (), audio=action.audio)

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
        # Deregister from the live-conversations registry so the scheduler
        # stops trying to inject into a closed session.
        live_conversations.deregister(stream_sid)

        # Resolve all pending tool confirms as denied
        for entry in pending_confirms.values():
            _, fut = entry
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
            agent.save_session(session_id)
            await agent.cleanup()

        if tts_pool:
            await tts_pool.stop()
        if stt:
            await stt.stop()

        # Save trace
        tracer.save(stream_sid)

        Logger.websocket_disconnected()
