"""
FastAPI server for AINow.

Endpoints:
- GET / - Browser voice UI
- GET /health - Health check
- WebSocket /ws/browser - Browser voice stream
- GET /trace/latest - Returns the most recent call trace as JSON
- GET /bench/ttft - Benchmark TTFT across OpenAI models
"""

import json
import os
import time
import asyncio
import random
import re
from collections import defaultdict
from pathlib import Path
import tempfile
from typing import List, Optional

from fastapi import FastAPI, WebSocket, Response, Query, Body, UploadFile, File, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from .conversation import run_conversation_over_browser
from .path_security import is_within_base, resolve_within_base
from .log import get_logger
from .services import agents as agent_store
from .services.scheduler import scheduler_service, is_valid_schedule, next_fire_times

logger = get_logger("ainow.server")

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

app = FastAPI(title="AINow", docs_url=None, redoc_url=None)


def _require_object_payload(payload, field_name: str = "payload"):
    if not isinstance(payload, dict):
        return JSONResponse({"error": f"{field_name} must be a JSON object"}, status_code=400)
    return None


def _safe_session_id(value: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    sid = value.strip()
    if not sid or not _SESSION_ID_RE.fullmatch(sid):
        return None
    return sid


@app.on_event("startup")
async def _startup_scheduler():
    await scheduler_service.start()


@app.on_event("startup")
async def _preload_whisper():
    """If we're in server-side STT mode (BROWSER_STT not set), pre-load the
    Whisper model so the first WebSocket connection doesn't pay the cold-start."""
    if os.getenv("BROWSER_STT"):
        return
    import asyncio as _asyncio
    from .services.whisper_stt import preload_whisper_model
    model_size = os.getenv("WHISPER_MODEL", "small")
    # Run in a thread so the load doesn't block the event loop startup
    _asyncio.create_task(_asyncio.to_thread(preload_whisper_model, model_size))


@app.on_event("startup")
async def _preload_kokoro():
    """If server-side Kokoro TTS is enabled, pre-load the model on startup so
    the first turn doesn't pay the cold-start (and we can verify GPU placement)."""
    if not os.getenv("SERVER_TTS") or not os.getenv("LOCAL_TTS_VOICE"):
        return
    import asyncio as _asyncio
    from .services.local_tts import _load_model as _load_kokoro
    _asyncio.create_task(_asyncio.to_thread(_load_kokoro))


@app.on_event("shutdown")
async def _shutdown_scheduler():
    await scheduler_service.stop()

# Serve static files (VAD libs, ONNX models, etc.)
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Graceful shutdown / connection draining ───────────────────────────
_draining = False          # Set True on SIGTERM — reject new connections
_active_calls = 0          # Count of live WebSocket conversations
_drain_event = asyncio.Event()  # Signalled when _active_calls hits 0

_ADMIN_HEADER = "x-ainow-admin-token"
_TRACE_DIR = Path(os.getenv("AINOW_TRACE_DIR", str(Path(tempfile.gettempdir()) / "ainow")))


def _resolve_agent_path(agent_name: str, rel_path: str) -> Path:
    base = Path(agent_store.agent_dir(agent_name)).resolve()
    return resolve_within_base(base, rel_path)


def _client_host(request: Request) -> str:
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or ""


def _has_admin_access(request: Request) -> bool:
    host = _client_host(request)
    if host in {"127.0.0.1", "::1", "localhost"}:
        return True
    token = os.getenv("AINOW_ADMIN_TOKEN", "").strip()
    if token and request.headers.get(_ADMIN_HEADER, "") == token:
        return True
    return False


def _require_admin(request: Request) -> Optional[JSONResponse]:
    if _has_admin_access(request):
        return None
    return JSONResponse(
        {
            "error": "admin access required",
            "hint": f"Use localhost or send {_ADMIN_HEADER} matching AINOW_ADMIN_TOKEN.",
        },
        status_code=403,
    )


def _require_debug_routes(request: Request) -> Optional[JSONResponse]:
    deny = _require_admin(request)
    if deny is not None:
        return deny
    if os.getenv("AINOW_ENABLE_DEBUG_ROUTES") == "1":
        return None
    return JSONResponse({"error": "debug routes are disabled"}, status_code=404)


@app.get("/")
async def browser_ui():
    """Serve the browser voice UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return Response(content=html_path.read_text(encoding="utf-8"), media_type="text/html")


@app.get("/api/config")
async def api_config():
    """Return initial config before WebSocket connect."""
    agent_store.ensure_default()
    return {
        "active_agent": agent_store.get_active(),
        "use_browser_tts": not bool(os.getenv("SERVER_TTS")),
        "use_browser_stt": bool(os.getenv("BROWSER_STT")),
    }


# ── Sessions (scoped to active agent) ─────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    from .services.llm import LLMService
    return LLMService.list_sessions(agent_store.get_active())


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    sid = _safe_session_id(session_id)
    if sid is None:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)
    path = agent_store.sessions_dir(agent_store.get_active()) / f"{sid}.json"
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed reading session file {path}: {e}")
        return JSONResponse({"error": "invalid session file"}, status_code=500)
    if not isinstance(data, dict):
        return JSONResponse({"error": "invalid session payload"}, status_code=500)
    messages = []
    raw_msgs = data.get("messages", [])
    if not isinstance(raw_msgs, list):
        logger.warning(f"Invalid session messages in {path}: expected list")
        raw_msgs = []
    # Pre-index tool results by tool_call_id so we can attach them to the
    # corresponding assistant tool_call entries.
    tool_results = {}
    for m in raw_msgs:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            tool_results[m["tool_call_id"]] = c
    for msg in raw_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        images: list = []
        if isinstance(content, list):
            text_parts: list = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type")
                if ctype == "text":
                    text_parts.append(c.get("text", ""))
                elif ctype == "image_url":
                    url = (c.get("image_url") or {}).get("url", "")
                    if isinstance(url, str) and url.startswith("data:"):
                        mime = ""
                        # Extract mime from data URL: data:<mime>;base64,...
                        if ";" in url[:50]:
                            mime = url[5:].split(";", 1)[0]
                        images.append({"data": url, "mime": mime})
                elif ctype in ("input_audio", "audio_url"):
                    if ctype == "input_audio":
                        audio = c.get("input_audio") or {}
                        fmt = audio.get("format", "mp3")
                        b64 = audio.get("data", "")
                        if b64:
                            images.append({"data": f"data:audio/{fmt};base64,{b64}", "mime": f"audio/{fmt}"})
                    else:
                        url = (c.get("audio_url") or {}).get("url", "")
                        if isinstance(url, str) and url.startswith("data:"):
                            mime = url[5:].split(";", 1)[0] if ";" in url[:50] else "audio/mpeg"
                            images.append({"data": url, "mime": mime})
            content = " ".join(text_parts)
        if role == "thinking":
            messages.append({
                "role": "thinking",
                "text": content,
                "duration": msg.get("duration", 0),
            })
        elif role == "assistant":
            if content:
                messages.append({"role": "assistant", "text": content})
            for tc in msg.get("tool_calls", []) or []:
                fn = (tc.get("function") or {})
                tcid = tc.get("id") or ""
                messages.append({
                    "role": "tool",
                    "toolName": fn.get("name", ""),
                    "toolArgs": fn.get("arguments", ""),
                    "toolResult": tool_results.get(tcid, ""),
                    "text": "",
                })
        elif role == "user" and (content or images):
            entry = {"role": "user", "text": content}
            if images:
                entry["images"] = images
            messages.append(entry)
        # role == "tool" handled via tool_results mapping above
    return {"id": data.get("id"), "messages": messages}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    sid = _safe_session_id(session_id)
    if sid is None:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)
    path = agent_store.sessions_dir(agent_store.get_active()) / f"{sid}.json"
    if path.exists():
        path.unlink()
        return {"deleted": True}
    return {"deleted": False}


# ── File Browser ─────────────────────────────────────────────────────

@app.get("/api/files/tree")
async def file_tree(path: str = Query("")):
    """List directory contents relative to the active agent's cwd."""
    agent_name = agent_store.get_active()
    base = Path(agent_store.agent_dir(agent_name)).resolve()
    try:
        target = _resolve_agent_path(agent_name, path)
    except PermissionError:
        return JSONResponse({"error": "access denied"}, status_code=403)
    if not target.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    if not target.is_dir():
        return JSONResponse({"error": "not a directory"}, status_code=400)
    entries = []
    try:
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            # Skip hidden files and __pycache__
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            rel = str(item.relative_to(base)).replace("\\", "/")
            entries.append({
                "name": item.name,
                "path": rel,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
    except PermissionError:
        pass
    return {"base": str(base), "path": path, "entries": entries}


@app.get("/api/files/read")
async def file_read(path: str = Query(...)):
    """Read a file relative to the active agent's cwd. Returns content + metadata."""
    agent_name = agent_store.get_active()
    base = Path(agent_store.agent_dir(agent_name)).resolve()
    try:
        target = _resolve_agent_path(agent_name, path)
    except PermissionError:
        return JSONResponse({"error": "access denied"}, status_code=403)
    if not target.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    if not target.is_file():
        return JSONResponse({"error": "not a file"}, status_code=400)
    # Limit to reasonable file sizes (1MB)
    if target.stat().st_size > 1_000_000:
        return JSONResponse({"error": "file too large"}, status_code=413)
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    ext = target.suffix.lower()
    lang_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
        ".jsx": "jsx", ".html": "html", ".css": "css", ".json": "json",
        ".md": "markdown", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
        ".sh": "bash", ".bash": "bash", ".rs": "rust", ".go": "go",
        ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
        ".rb": "ruby", ".php": "php", ".sql": "sql", ".xml": "xml",
    }
    return {
        "path": path,
        "name": target.name,
        "language": lang_map.get(ext, ""),
        "content": content,
        "size": target.stat().st_size,
        "lines": content.count("\n") + 1,
    }


@app.get("/api/files/raw")
async def file_raw(path: str = Query(...)):
    """Stream a file's bytes (for images/audio/video) sandboxed to active agent dir."""
    from fastapi.responses import FileResponse
    import mimetypes
    agent_name = agent_store.get_active()
    base = Path(agent_store.agent_dir(agent_name)).resolve()
    try:
        target = _resolve_agent_path(agent_name, path)
    except PermissionError:
        return JSONResponse({"error": "access denied"}, status_code=403)
    if not target.exists() or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    mime, _ = mimetypes.guess_type(str(target))
    return FileResponse(str(target), media_type=mime or "application/octet-stream")


@app.post("/api/sessions/{session_id}/title")
async def generate_session_title(session_id: str, payload: dict = Body(...)):
    """Generate an AI title for a session, or set one manually."""
    bad_payload = _require_object_payload(payload, "payload")
    if bad_payload is not None:
        return bad_payload
    sid = _safe_session_id(session_id)
    if sid is None:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)
    path = agent_store.sessions_dir(agent_store.get_active()) / f"{sid}.json"
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    if "title" in payload and not isinstance(payload.get("title"), str):
        return JSONResponse({"error": "title must be a string"}, status_code=400)

    manual_title = (payload.get("title") or "").strip()
    if manual_title:
        # Manual title set
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return JSONResponse({"error": f"failed to read session: {e}"}, status_code=500)
        if not isinstance(data, dict):
            return JSONResponse({"error": "invalid session payload"}, status_code=500)
        data["title"] = manual_title
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return JSONResponse({"error": f"failed to write session: {e}"}, status_code=500)
        return {"title": manual_title}

    # Auto-generate from first user message using LLM
    if "message" not in payload or not isinstance(payload.get("message"), str):
        return JSONResponse({"error": "message is required and must be a string"}, status_code=400)
    message = (payload.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "no message"}, status_code=400)

    try:
        from openai import AsyncOpenAI as _OAI
        base_url = os.getenv("LLM_BASE_URL", f"http://localhost:{os.getenv('LLAMA_SERVER_PORT', '8080')}/v1")
        api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "not-needed"))
        client = _OAI(base_url=base_url, api_key=api_key)
        resp = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "local"),
            messages=[{
                "role": "user",
                "content": f"Generate a very short title (3-6 words, no quotes, no punctuation at the end) for a conversation that starts with this message:\n\n{message[:300]}",
            }],
            max_tokens=20,
            temperature=0.7,
        )
        title = (resp.choices[0].message.content or "").strip().strip('"\'').strip()
    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        # Fallback: truncate the message
        title = message[:40] + ("..." if len(message) > 40 else "")

    # Save title to session file
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return JSONResponse({"error": "invalid session payload"}, status_code=500)
        data["title"] = title
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed saving generated title for {sid}: {e}")

    return {"title": title}


# ── Agents ────────────────────────────────────────────────────────────

@app.get("/api/agents")
async def list_agents():
    agent_store.ensure_default()
    return agent_store.list_agents()


@app.post("/api/agents")
async def create_agent(payload: dict = Body(...)):
    bad_payload = _require_object_payload(payload, "payload")
    if bad_payload is not None:
        return bad_payload
    name = (payload.get("name") or "").strip()
    claude_md = payload.get("claude_md", "")
    try:
        agent_store.create(name, claude_md)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"name": name, "active": agent_store.get_active() == name}


@app.get("/api/agents/{name}")
async def get_agent(name: str):
    if not agent_store.exists(name):
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "name": name,
        "active": agent_store.get_active() == name,
        "claude_md": agent_store.read_claude_md(name),
        "mcp_servers": agent_store.read_mcp_servers(name),
    }


@app.put("/api/agents/{name}")
async def update_agent(name: str, payload: dict = Body(...)):
    bad_payload = _require_object_payload(payload, "payload")
    if bad_payload is not None:
        return bad_payload
    if not agent_store.exists(name):
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        if "claude_md" in payload:
            agent_store.write_claude_md(name, payload.get("claude_md", ""))
        if "mcp_servers" in payload:
            servers = payload.get("mcp_servers") or {}
            if not isinstance(servers, dict):
                return JSONResponse({"error": "mcp_servers must be an object"}, status_code=400)
            agent_store.write_mcp_servers(name, servers)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"saved": True}


@app.delete("/api/agents/{name}")
async def delete_agent(name: str):
    try:
        agent_store.delete(name)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"deleted": True}


@app.post("/api/agents/{name}/activate")
async def activate_agent(name: str):
    try:
        agent_store.set_active(name)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"active": name}


# ── Agent import / export ─────────────────────────────────────────────

# Paths excluded from export (not portable / large / local-only)
_EXPORT_EXCLUDE = {"sessions", "scheduled_tasks.json", "node_modules",
                   ".cache", "__pycache__", ".task-scheduler.pid",
                   ".task-scheduler.log", ".task-scheduler-data.json"}
_MAX_IMPORT_ARCHIVE_BYTES = 64 * 1024 * 1024  # 64 MB
_MAX_IMPORT_MEMBERS = 2000
_MAX_MEMBER_PATH_LEN = 260
_MAX_MEMBER_FILE_BYTES = 5 * 1024 * 1024  # 5 MB per file


@app.get("/api/agents/{name}/export")
async def export_agent(name: str):
    """Download an agent as a tar.gz archive."""
    import io, tarfile
    try:
        agent_store._validate_name(name)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not agent_store.exists(name):
        return JSONResponse({"error": "agent not found"}, status_code=404)

    agent_dir = agent_store.agent_dir(name)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(str(agent_dir)):
            # Prune excluded dirs in-place so os.walk skips them
            dirs[:] = [d for d in dirs if d not in _EXPORT_EXCLUDE]
            for f in files:
                if f in _EXPORT_EXCLUDE:
                    continue
                full = os.path.join(root, f)
                if os.path.islink(full):
                    continue
                arcname = os.path.join(name, os.path.relpath(full, str(agent_dir)))
                tar.add(full, arcname=arcname)
    buf.seek(0)
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"agent-export-{name}-{ts}.tar.gz"
    return StreamingResponse(
        buf,
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/agents/import")
async def import_agent(request: Request, file: UploadFile = File(...)):
    """Import an agent from a tar.gz archive.

    The archive must contain a single top-level directory (the agent name).
    If the agent already exists, its files are merged (new files overwrite).
    """
    deny = _require_admin(request)
    if deny is not None:
        return deny

    import io, tarfile, posixpath
    if not file.filename or not file.filename.endswith((".tar.gz", ".tgz")):
        return JSONResponse({"error": "file must be a .tar.gz"}, status_code=400)

    data = await file.read(_MAX_IMPORT_ARCHIVE_BYTES + 1)
    if len(data) > _MAX_IMPORT_ARCHIVE_BYTES:
        return JSONResponse({"error": "archive too large"}, status_code=413)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            # Determine the agent name from the top-level directory and validate entries.
            members = tar.getmembers()
            if len(members) > _MAX_IMPORT_MEMBERS:
                return JSONResponse({"error": f"archive contains too many members (max {_MAX_IMPORT_MEMBERS})"}, status_code=400)

            def _normalize_member_name(raw_name: str) -> str:
                if not raw_name:
                    raise ValueError("empty member name")
                posix_name = raw_name.replace("\\", "/").strip()
                if posix_name.endswith("/"):
                    posix_name = posix_name.rstrip("/")
                if not posix_name:
                    raise ValueError("empty member name")
                if ":" in posix_name:
                    raise ValueError(f"invalid character ':' in member: {raw_name}")
                if posixpath.isabs(posix_name):
                    raise ValueError(f"absolute path not allowed: {raw_name}")
                if posix_name.startswith("..") or "/../" in posix_name or posix_name.endswith("/.."):
                    raise ValueError(f"path traversal detected: {raw_name}")
                if len(posix_name) > _MAX_MEMBER_PATH_LEN:
                    raise ValueError(f"member path too long: {raw_name}")
                parts = posix_name.split("/")
                if any(part in {"", ".", ".."} for part in parts):
                    raise ValueError(f"invalid path segment in member: {raw_name}")
                return posix_name

            normalized_members = []
            top_dirs = set()
            for m in members:
                normalized = _normalize_member_name(m.name)
                if m.issym() or m.islnk():
                    raise ValueError(f"symlink entry blocked: {m.name}")
                parts = normalized.split("/")
                if parts:
                    top_dirs.add(parts[0])
                normalized_members.append((m, normalized))

            if len(top_dirs) != 1:
                return JSONResponse({
                    "error": f"archive must contain exactly one top-level folder (found: {top_dirs})"
                }, status_code=400)

            agent_name = next(iter(top_dirs))
            # Validate agent name
            try:
                agent_store._validate_name(agent_name)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)

            # Create agent dir if needed (don't fail if already exists)
            dest = agent_store.agent_dir(agent_name)
            dest.mkdir(parents=True, exist_ok=True)
            agent_store.sessions_dir(agent_name).mkdir(exist_ok=True)

            # Extract all files into agents/<name>/
            extracted = 0
            for m, normalized in normalized_members:
                if m.isdir():
                    # Keep directory entries aligned; skip empty dirs after target creation.
                    target_dir = resolve_within_base(dest, normalized)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    continue
                if not m.isfile():
                    logger.warning(f"Skipping unsupported tar member type: {m.name}")
                    continue
                rel = normalized[len(agent_name) + 1:] if normalized.startswith(f"{agent_name}/") else ""
                if not rel:
                    continue
                target = resolve_within_base(dest, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    f = tar.extractfile(m)
                    if not f:
                        return JSONResponse({"error": f"could not read member: {m.name}"}, status_code=400)
                    payload = f.read(_MAX_MEMBER_FILE_BYTES + 1)
                except Exception as e:
                    logger.error(f"Failed importing member {m.name}: {e}")
                    return JSONResponse({"error": f"failed to extract {m.name}"}, status_code=400)
                if len(payload) > _MAX_MEMBER_FILE_BYTES:
                    return JSONResponse({"error": f"member too large: {m.name}"}, status_code=413)
                target.write_bytes(payload)
                extracted += 1

            # Map the archive's metadata.json to AINow's meta.json if needed
            archive_meta_path = dest / "metadata.json"
            ainow_meta_path = agent_store.meta_path(agent_name)
            if archive_meta_path.exists() and not ainow_meta_path.exists():
                # Bootstrap meta.json from the archive's metadata
                try:
                    import json as _json
                    ext_meta = _json.loads(archive_meta_path.read_text(encoding="utf-8"))
                    if not isinstance(ext_meta, dict):
                        raise ValueError("metadata.json must be an object")
                    display_name = (ext_meta.get("employee", {}).get("custom_name")
                                   or agent_name)
                    lang = ext_meta.get("employee", {}).get("default_language") or "en"
                    from datetime import datetime
                    ainow_meta = {
                        "display_name": display_name,
                        "created_at": ext_meta.get("exportDate", datetime.now().isoformat()),
                        "mcp_servers": {},
                        "preferences": {"lang": f"{lang}-{'US' if lang == 'en' else lang.upper()}"},
                    }
                    ainow_meta_path.write_text(
                        _json.dumps(ainow_meta, indent=2), encoding="utf-8"
                    )
                except Exception as e:
                    logger.error(f"Failed to map imported metadata for {agent_name}: {e}")
            elif not ainow_meta_path.exists():
                # No metadata at all — write a minimal meta.json
                agent_store._write_meta(agent_name)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except PermissionError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return JSONResponse({"error": "import failed"}, status_code=500)

    # Auto-patch: if the imported agent has a memory-db skill using the old
    # Linux-only @sqliteai/sqlite-vector extension, replace its core files
    # with our cross-platform sqlite-vec version.
    patched_skill = False
    imported_pkg = dest / ".skills" / "memory-db" / "package.json"
    if imported_pkg.exists():
        try:
            pkg_text = imported_pkg.read_text(encoding="utf-8")
            if "@sqliteai/sqlite-vector" in pkg_text:
                patches_dir = Path(__file__).parent / "patches" / "memory-db"
                if patches_dir.is_dir():
                    for patch_file in ["cli.js", "package.json", "skill.md"]:
                        src = patches_dir / patch_file
                        if src.is_file():
                            (dest / ".skills" / "memory-db" / patch_file).write_bytes(
                                src.read_bytes()
                            )
                    patched_skill = True
                    logger.info(f"Auto-patched memory-db skill for agent '{agent_name}' (sqlite-vec)")
        except Exception as e:
            logger.error(f"Failed to auto-patch memory-db: {e}")

    return {
        "name": agent_name,
        "files_extracted": extracted,
        "patched_memory_db": patched_skill,
    }


# ── Scheduled tasks ───────────────────────────────────────────────────

@app.get("/api/agents/{name}/scheduled_tasks")
async def list_scheduled_tasks(name: str):
    if not agent_store.exists(name):
        return JSONResponse({"error": "agent not found"}, status_code=404)
    return {"tasks": agent_store.read_scheduled_tasks(name)}


@app.post("/api/agents/{name}/scheduled_tasks")
async def create_scheduled_task(name: str, payload: dict = Body(...)):
    bad_payload = _require_object_payload(payload, "payload")
    if bad_payload is not None:
        return bad_payload
    if not agent_store.exists(name):
        return JSONResponse({"error": "agent not found"}, status_code=404)
    task_name = (payload.get("name") or "").strip()
    schedule = (payload.get("schedule") or "").strip()
    prompt = payload.get("prompt") or ""
    mode = payload.get("mode") or "new_session"
    enabled = bool(payload.get("enabled", True))
    if not task_name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    if not schedule:
        return JSONResponse({"error": "schedule is required"}, status_code=400)
    if not is_valid_schedule(schedule):
        return JSONResponse({"error": "invalid schedule (must be cron or YYYY-MM-DD HH:MM)"}, status_code=400)
    if not prompt.strip():
        return JSONResponse({"error": "prompt is required"}, status_code=400)
    if mode not in ("new_session", "inject"):
        return JSONResponse({"error": "mode must be 'new_session' or 'inject'"}, status_code=400)
    task = agent_store.add_scheduled_task(name, {
        "name": task_name,
        "schedule": schedule,
        "prompt": prompt,
        "mode": mode,
        "enabled": enabled,
    })
    scheduler_service.reload_agent(name)
    return task


@app.put("/api/agents/{name}/scheduled_tasks/{task_id}")
async def update_scheduled_task_route(name: str, task_id: str, payload: dict = Body(...)):
    bad_payload = _require_object_payload(payload, "payload")
    if bad_payload is not None:
        return bad_payload
    if not agent_store.exists(name):
        return JSONResponse({"error": "agent not found"}, status_code=404)
    patch = {}
    if "name" in payload:
        v = (payload.get("name") or "").strip()
        if not v:
            return JSONResponse({"error": "name cannot be empty"}, status_code=400)
        patch["name"] = v
    if "schedule" in payload:
        v = (payload.get("schedule") or "").strip()
        if not is_valid_schedule(v):
            return JSONResponse({"error": "invalid schedule"}, status_code=400)
        patch["schedule"] = v
    if "prompt" in payload:
        patch["prompt"] = payload.get("prompt") or ""
    if "mode" in payload:
        m = payload.get("mode")
        if m not in ("new_session", "inject"):
            return JSONResponse({"error": "mode must be 'new_session' or 'inject'"}, status_code=400)
        patch["mode"] = m
    if "enabled" in payload:
        patch["enabled"] = bool(payload.get("enabled"))
    updated = agent_store.update_scheduled_task(name, task_id, patch)
    if not updated:
        return JSONResponse({"error": "task not found"}, status_code=404)
    scheduler_service.reload_agent(name)
    return updated


@app.delete("/api/agents/{name}/scheduled_tasks/{task_id}")
async def delete_scheduled_task_route(name: str, task_id: str):
    if not agent_store.exists(name):
        return JSONResponse({"error": "agent not found"}, status_code=404)
    if not agent_store.delete_scheduled_task(name, task_id):
        return JSONResponse({"error": "task not found"}, status_code=404)
    scheduler_service.reload_agent(name)
    return {"deleted": True}


@app.post("/api/agents/{name}/scheduled_tasks/{task_id}/run")
async def run_scheduled_task_now(name: str, task_id: str):
    if not agent_store.exists(name):
        return JSONResponse({"error": "agent not found"}, status_code=404)
    result = await scheduler_service.run_now(name, task_id)
    return result


@app.get("/api/cron/next")
async def cron_next(expr: str = Query(...), n: int = Query(3, ge=1, le=10)):
    """Return up to n upcoming fire times for a schedule (cron or ISO)."""
    if not is_valid_schedule(expr):
        return {"valid": False, "next": []}
    fires = next_fire_times(expr, n=n)
    return {"valid": True, "next": [f.isoformat() for f in fires]}


@app.get("/api/agents/{name}/profile")
async def agent_profile(name: str):
    """Serve the agent's profile image (profile.jpg/png) if present."""
    if not agent_store.exists(name):
        return JSONResponse({"error": "not found"}, status_code=404)
    d = agent_store.agent_dir(name)
    for fname in ("profile.jpg", "profile.jpeg", "profile.png", "profile.webp"):
        p = d / fname
        if p.is_file():
            mime = "image/jpeg" if fname.endswith((".jpg", ".jpeg")) else (
                "image/png" if fname.endswith(".png") else "image/webp"
            )
            return Response(content=p.read_bytes(), media_type=mime)
    return JSONResponse({"error": "no profile image"}, status_code=404)


@app.get("/api/models")
async def list_models():
    """List available models."""
    from .services.model_manager import MODELS, MODEL_ALIASES, model_manager
    active_model = model_manager.current_model
    vision_enabled = model_manager.vision_enabled
    ctx_active = model_manager.context_size
    models = []
    for alias, model_id in MODEL_ALIASES.items():
        config = MODELS.get(model_id, {})
        has_mmproj = bool(config.get("mmproj"))
        if config.get("online"):
            # Online model is "active" when llama-server is stopped and the
            # env vars point to this provider (= user switched to online).
            is_active = (
                not active_model
                and os.environ.get("LLM_MODEL") == config.get("model_id")
            )
        else:
            is_active = (active_model == model_id)
        models.append({
            "alias": alias,
            "id": model_id,
            "name": config.get("name", alias),
            "online": config.get("online", False),
            "active": is_active,
            "has_mmproj": has_mmproj,
            "vision_enabled": (vision_enabled if is_active else True) if has_mmproj else False,
            "ctx": ctx_active if is_active else 0,
        })
    return models


@app.get("/api/runtime")
async def get_runtime(request: Request):
    """Return the current runtime state."""
    deny = _require_admin(request)
    if deny is not None:
        return deny

    from .services.model_manager import model_manager
    return {
        "current_model": model_manager.current_model,
        "vision_enabled": model_manager.vision_enabled,
        "context_size": model_manager.context_size,
        "thinking_enabled": model_manager.thinking_enabled,
    }


@app.post("/api/runtime/settings")
async def set_runtime_settings(request: Request, payload: dict = Body(...)):
    """Update runtime settings (vision, ctx) for the active local model.

    Body: { "vision_enabled"?: bool, "ctx"?: int }
    Only provided fields are updated. Changes are persisted to the active
    agent's preferences and trigger ONE llama-server reload.
    """
    deny = _require_admin(request)
    if deny is not None:
        return deny
    bad_payload = _require_object_payload(payload, "payload")
    if bad_payload is not None:
        return bad_payload

    from .services.model_manager import model_manager, MODELS
    model_id = model_manager.current_model
    if not model_id:
        return JSONResponse({"error": "no local model is currently active"}, status_code=400)
    if model_id not in MODELS:
        return JSONResponse({"error": f"unknown model: {model_id}"}, status_code=400)
    config = MODELS[model_id]
    if config.get("online"):
        return JSONResponse({"error": "runtime settings only apply to local models"}, status_code=400)

    # Read the existing prefs, merge the patch
    try:
        active_agent = agent_store.get_active()
        prefs = dict(agent_store.read_preferences(active_agent))
    except Exception:
        prefs = {}

    patch = {}
    if "vision_enabled" in payload:
        if not config.get("mmproj"):
            return JSONResponse({"error": f"model '{model_id}' has no mmproj"}, status_code=400)
        patch["vision_enabled"] = bool(payload["vision_enabled"])
    if "ctx" in payload:
        try:
            ctx_val = int(payload["ctx"])
            if ctx_val < 512:
                return JSONResponse({"error": "ctx must be >= 512"}, status_code=400)
            patch["ctx"] = ctx_val
        except (TypeError, ValueError):
            return JSONResponse({"error": "ctx must be an integer"}, status_code=400)
    if "thinking_enabled" in payload:
        patch["thinking_enabled"] = bool(payload["thinking_enabled"])

    if not patch:
        return JSONResponse({"error": "no settings provided"}, status_code=400)

    # Persist to the active agent's preferences
    try:
        agent_store.update_preferences(active_agent, patch)
    except Exception as e:
        logger.error(f"Failed to persist runtime settings: {e}")

    # Compute effective values (merged) for the reload
    merged = {**prefs, **patch}
    vision_enabled = bool(merged.get("vision_enabled", True))
    thinking_enabled = bool(merged.get("thinking_enabled", False))
    ctx_override = merged.get("ctx")
    try:
        if ctx_override is not None:
            ctx_override = int(ctx_override)
    except Exception:
        ctx_override = None

    # Reload llama-server with the new flags
    try:
        import asyncio
        await asyncio.to_thread(
            model_manager.start, model_id, vision_enabled, ctx_override,
            thinking_enabled,
        )
    except Exception as e:
        return JSONResponse({"error": f"reload failed: {e}"}, status_code=500)

    return {
        "model_id": model_id,
        "vision_enabled": vision_enabled,
        "thinking_enabled": thinking_enabled,
        "ctx": model_manager.context_size,
        "reloaded": True,
    }


@app.post("/api/eject-model")
async def eject_model(request: Request):
    """Eject the current model — stop llama-server and free VRAM."""
    deny = _require_admin(request)
    if deny is not None:
        return deny

    from .services.model_manager import model_manager
    try:
        import asyncio
        await asyncio.to_thread(model_manager.stop)
        return {"ejected": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/models/{alias}")
async def switch_model(alias: str, request: Request):
    """Switch to a different local model (restarts llama-server)."""
    deny = _require_admin(request)
    if deny is not None:
        return deny

    from .services.model_manager import resolve_model_id, MODELS, model_manager
    try:
        model_id = resolve_model_id(alias)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    config = MODELS[model_id]
    if config.get("online"):
        # Online models — set env vars so next LLM connection uses them
        api_key = os.getenv(config.get("api_key_env", ""), "")
        if not api_key:
            return JSONResponse({"error": f"Missing {config.get('api_key_env', '')} in .env"}, status_code=400)
        os.environ["LLM_BASE_URL"] = config["base_url"]
        os.environ["LLM_API_KEY"] = api_key
        os.environ["LLM_MODEL"] = config["model_id"]
        # Free GPU memory: kill the local llama-server if one is running
        try:
            import asyncio
            await asyncio.to_thread(model_manager.stop)
        except Exception as e:
            logger.error(f"Failed to stop local model on online switch: {e}")
        # Persist "online" as the active agent's preferred model
        try:
            agent_store.update_preferences(agent_store.get_active(), {"model": model_id})
        except Exception:
            pass
        return {
            "switched": True,
            "model": model_id,
            "online": True,
            "name": config.get("name", model_id),
        }

    # Local model — restart llama-server (blocking)
    import asyncio
    # Pull vision/ctx overrides from the active agent's preferences
    try:
        active_agent = agent_store.get_active()
        prefs = agent_store.read_preferences(active_agent)
    except Exception:
        prefs = {}
    vision_enabled = bool(prefs.get("vision_enabled", True))
    thinking_enabled = bool(prefs.get("thinking_enabled", False))
    ctx_override = prefs.get("ctx")
    try:
        if ctx_override is not None:
            ctx_override = int(ctx_override)
    except Exception:
        ctx_override = None
    try:
        await asyncio.to_thread(
            model_manager.start, model_id, vision_enabled, ctx_override,
            thinking_enabled,
        )
        os.environ["LLM_MODEL"] = model_id
        os.environ["LLM_BASE_URL"] = f"http://localhost:{os.getenv('LLAMA_SERVER_PORT', '8080')}/v1"
        os.environ["LLM_API_KEY"] = "not-needed"
        # Persist the model choice to the active agent's prefs
        try:
            agent_store.update_preferences(active_agent, {"model": model_id})
        except Exception:
            pass
        return {"switched": True, "model": model_id, "name": config.get("name", alias)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/test-thinking")
async def test_thinking(request: Request):
    """Debug: test reasoning capture inside the actual server process."""
    deny = _require_debug_routes(request)
    if deny is not None:
        return deny

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="x")
    rc = 0; cc = 0
    stream = await client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=200, stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta: continue
        extras = getattr(delta, "model_extra", None) or {}
        if extras.get("reasoning_content"): rc += 1
        if delta.content: cc += 1
    return {"reasoning_chunks": rc, "content_chunks": cc}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/trace/latest")
async def latest_trace(request: Request):
    """Return the most recent call trace as JSON."""
    deny = _require_debug_routes(request)
    if deny is not None:
        return deny

    trace_dir = _TRACE_DIR
    if not trace_dir.exists():
        return JSONResponse({"error": "No traces found"}, status_code=404)

    traces = sorted(trace_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not traces:
        return JSONResponse({"error": "No traces found"}, status_code=404)

    data = json.loads(traces[0].read_text())
    return JSONResponse(data)


## ── TTFT Benchmark ──────────────────────────────────────────────

BENCH_PROMPT = "Explain how a combustion engine works."

# Each entry: (display_name, provider_key, model_id)
# provider_key is used to look up the right AsyncOpenAI client
DEFAULT_MODELS = [
    # OpenAI 4-series
    ("gpt-4o-mini",   "openai", "gpt-4o-mini"),
    ("gpt-4o",        "openai", "gpt-4o"),
    ("gpt-4.1-nano",  "openai", "gpt-4.1-nano"),
    ("gpt-4.1-mini",  "openai", "gpt-4.1-mini"),
    ("gpt-4.1",       "openai", "gpt-4.1"),
    # OpenAI 5-series
    ("gpt-5-nano",    "openai", "gpt-5-nano"),
    ("gpt-5-mini",    "openai", "gpt-5-mini"),
    ("gpt-5",         "openai", "gpt-5"),
    ("gpt-5.1",       "openai", "gpt-5.1"),
    ("gpt-5.2",       "openai", "gpt-5.2"),
    # Groq
    ("groq/llama-3.3-70b",  "groq", "llama-3.3-70b-versatile"),
    ("groq/llama-3.1-8b",   "groq", "llama-3.1-8b-instant"),
]

BENCH_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": BENCH_PROMPT},
]


def _make_clients() -> dict:
    """Build provider -> AsyncOpenAI client map."""
    clients = {}
    oai_key = os.getenv("OPENAI_API_KEY", "")
    if oai_key:
        clients["openai"] = AsyncOpenAI(api_key=oai_key)
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        clients["groq"] = AsyncOpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
    return clients


async def _measure_ttft(client: AsyncOpenAI, model: str) -> float:
    """
    Single TTFT measurement in milliseconds.

    Opens a streaming completion, records time-to-first-content-token,
    then closes the stream immediately.
    """
    # GPT-5+ uses max_completion_tokens; older models use max_tokens
    is_new = model.startswith(("gpt-5", "o1", "o3", "o4"))
    token_param = "max_completion_tokens" if is_new else "max_tokens"

    params: dict = {
        "model": model,
        "messages": BENCH_MESSAGES,
        "stream": True,
        token_param: 20,
    }
    if is_new:
        # Use lowest reasoning effort the model accepts:
        # try "none" first, fall back to "minimal"
        params["extra_body"] = {"reasoning_effort": "none"}
    else:
        params["temperature"] = 0

    t0 = time.perf_counter()
    try:
        stream = await client.chat.completions.create(**params)
    except Exception as e:
        if is_new and "none" in str(e).lower():
            # Model doesn't support "none" -- retry with "minimal"
            params["extra_body"] = {"reasoning_effort": "minimal"}
            t0 = time.perf_counter()
            stream = await client.chat.completions.create(**params)
        else:
            raise
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            ttft_ms = (time.perf_counter() - t0) * 1000
            await stream.close()
            return ttft_ms
    # edge case: no content tokens at all
    return (time.perf_counter() - t0) * 1000



@app.get("/bench/ttft")
async def bench_ttft(
    request: Request,
    models: Optional[str] = Query(
        None,
        description="Comma-separated model names. Defaults to a built-in list.",
    ),
    runs: int = Query(30, ge=1, le=100, description="Runs per model"),
):
    """
    Benchmark TTFT across OpenAI-compatible models.

    Usage:
        curl https://your-server/bench/ttft
        curl https://your-server/bench/ttft?models=gpt-4o-mini,gpt-4o&runs=5
    """
    deny = _require_debug_routes(request)
    if deny is not None:
        return deny

    clients = _make_clients()

    # Build model list: use DEFAULT_MODELS or parse comma-separated overrides
    if models:
        # For custom input, assume openai provider unless "groq/" prefixed
        entries = []
        for m in models.split(","):
            m = m.strip()
            if not m:
                continue
            if m.startswith("groq/"):
                entries.append((m, "groq", m.removeprefix("groq/")))
            else:
                entries.append((m, "openai", m))
        model_entries = entries
    else:
        model_entries = DEFAULT_MODELS

    # Filter out models whose provider has no API key
    model_entries = [(name, prov, mid) for name, prov, mid in model_entries if prov in clients]

    # Build a shuffled schedule: each model appears `runs` times, interleaved
    schedule = [(name, prov, mid, i) for name, prov, mid in model_entries for i in range(runs)]
    random.shuffle(schedule)

    total = len(schedule)
    names = [name for name, _, _ in model_entries]
    logger.info(f"TTFT benchmark: {len(model_entries)} models x {runs} runs = {total} calls (randomised)")

    times_by_model: dict[str, list[float]] = defaultdict(list)
    errors_by_model: dict[str, list[str]] = defaultdict(list)

    for idx, (name, prov, mid, run_i) in enumerate(schedule, 1):
        try:
            ms = await _measure_ttft(clients[prov], mid)
            times_by_model[name].append(round(ms, 1))
            logger.info(f"  [{idx}/{total}] {name} #{run_i+1} -> {ms:.0f} ms")
        except Exception as e:
            errors_by_model[name].append(f"run {run_i+1}: {e}")
            logger.info(f"  [{idx}/{total}] {name} #{run_i+1} -> ERROR")

    # Aggregate stats per model (preserve original order)
    results = []
    for name in names:
        t = times_by_model.get(name, [])
        errs = errors_by_model.get(name, [])
        if not t:
            results.append({"model": name, "error": errs[0] if errs else "no data"})
            logger.info(f"  {name} -> ERROR: {errs[0] if errs else 'no data'}")
            continue
        avg = round(sum(t) / len(t), 1)
        entry: dict = {
            "model": name,
            "runs": len(t),
            "avg_ms": avg,
            "min_ms": min(t),
            "max_ms": max(t),
            "all_ms": t,
        }
        if errs:
            entry["errors"] = errs
        results.append(entry)
        logger.info(f"  {name} -> avg {avg} ms  (min {min(t)}, max {max(t)})")

    return JSONResponse({
        "prompt": BENCH_PROMPT,
        "runs_per_model": runs,
        "results": results,
    })


@app.websocket("/ws/browser")
async def browser_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for browser-based voice.

    Handles bidirectional audio: binary frames (PCM linear16 @ 16kHz).
    """
    global _active_calls

    if _draining:
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Server draining")
        return

    await websocket.accept()
    _active_calls += 1
    logger.info(f"Browser connected  (active: {_active_calls})")

    try:
        await run_conversation_over_browser(websocket)
    except Exception as e:
        logger.error(f"Browser WebSocket error: {e}")
    finally:
        _active_calls -= 1
        logger.info(f"Browser ended  (active: {_active_calls})")
        if _draining and _active_calls <= 0:
            _drain_event.set()
