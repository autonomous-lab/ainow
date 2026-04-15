"""
MCP (Model Context Protocol) integration.

Each agent declares its MCP servers in `agents/<name>/meta.json` under
`mcp_servers`:

  {
    "mcp_servers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
      },
      "github": {
        "command": "uvx",
        "args": ["mcp-server-github"],
        "env": {"GITHUB_TOKEN": "..."}
      }
    }
  }

Architecture note (important):
  The MCP Python SDK uses anyio task groups under the hood. `stdio_client`
  spawns a subprocess inside its own task group; the subprocess only stays
  alive as long as the *task that entered the `async with` block* is alive.
  We can't use AsyncExitStack from a transient caller and keep the
  subprocess alive afterwards — the parent task scope cancels the children
  on exit. (You'll see CancelledError at `_make_subprocess_transport` if
  you try.)

  So each MCP server runs inside its own long-lived asyncio.Task. The task
  enters `async with stdio_client(...)` + `async with ClientSession(...)`
  and then sits in a queue loop forever. Tool calls are dispatched via an
  asyncio.Queue with a future per call. Stop is signaled by enqueuing a
  sentinel.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from ..log import get_logger
from . import agents
from .tools import ToolDef, register_tool, unregister_tool

logger = get_logger("ainow.mcp")

# Sentinel pushed into the command queue to ask the worker to shut down.
_SHUTDOWN = object()


class MCPServerHandle:
    """One MCP server: long-lived background task owning a stdio session."""

    def __init__(self, name: str, config: dict, agent_name: Optional[str] = None):
        self.name = name
        self.config = config
        self.agent_name = agent_name  # used to anchor the subprocess cwd
        self.tools: List[Any] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ready = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._error: Optional[Exception] = None
        self._stopped = asyncio.Event()

    async def start(self, timeout: float = 30.0) -> None:
        """Spawn the worker task and wait until the session is ready (or fails)."""
        self._task = asyncio.create_task(self._worker(), name=f"mcp-{self.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            await self.stop()
            raise RuntimeError(f"MCP server '{self.name}' did not start within {timeout}s")
        if self._error:
            err = self._error
            self._error = None
            raise err

    async def call(self, tool_name: str, args: dict, timeout: float = 60.0) -> str:
        """Forward a tool call to the worker via the command queue."""
        if not self._task or self._task.done():
            return f"Error: MCP server '{self.name}' is not running"
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put(("call", tool_name, args, future))
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return f"Error: MCP {self.name}/{tool_name} timed out after {timeout}s"
        except Exception as e:
            return f"Error from MCP {self.name}/{tool_name}: {e}"
        return _format_mcp_content(result)

    async def stop(self) -> None:
        """Ask the worker to shut down cleanly. Idempotent."""
        if not self._task:
            return
        if not self._task.done():
            try:
                self._queue.put_nowait((_SHUTDOWN, None, None, None))
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None

    async def _worker(self) -> None:
        """The long-lived task that owns the stdio_client + ClientSession contexts."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            command = self.config.get("command")
            args = self.config.get("args", []) or []
            env = self.config.get("env") or None
            if not command:
                self._error = ValueError(f"MCP server '{self.name}' missing 'command'")
                self._ready.set()
                return

            merged_env = dict(os.environ)
            if env:
                merged_env.update(env)

            # Anchor the subprocess cwd to the owning agent's folder so any
            # relative paths the server uses (e.g. SQLite --db-path ./database.db,
            # Filesystem root, Memory KG file) land inside agents/<name>/.
            spawn_cwd: Optional[str] = None
            if self.agent_name:
                try:
                    spawn_cwd = str(agents.agent_dir(self.agent_name))
                except Exception:
                    spawn_cwd = None

            params = StdioServerParameters(
                command=command,
                args=args,
                env=merged_env,
                cwd=spawn_cwd,
            )

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self.tools = listed.tools or []
                    logger.info(f"MCP server '{self.name}' ready ({len(self.tools)} tools)")
                    self._ready.set()

                    # Command loop — exits on _SHUTDOWN sentinel
                    while True:
                        cmd = await self._queue.get()
                        kind = cmd[0]
                        if kind is _SHUTDOWN:
                            break
                        if kind == "call":
                            _, tool_name, call_args, fut = cmd
                            try:
                                result = await session.call_tool(tool_name, call_args)
                                if not fut.done():
                                    fut.set_result(result)
                            except Exception as e:
                                if not fut.done():
                                    fut.set_exception(e)
        except asyncio.CancelledError:
            # Task was cancelled externally — propagate cleanly
            raise
        except Exception as e:
            logger.error(f"MCP server '{self.name}' worker crashed: {e}")
            self._error = e
            if not self._ready.is_set():
                self._ready.set()
        finally:
            self._stopped.set()


def _format_mcp_content(result: Any) -> str:
    """Convert an MCP CallToolResult to a string for the LLM tool message."""
    parts: List[str] = []
    is_error = getattr(result, "isError", False)
    content = getattr(result, "content", []) or []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text" or hasattr(block, "text"):
            text = getattr(block, "text", "") or ""
            parts.append(text)
        elif block_type == "image" or hasattr(block, "data"):
            mime = getattr(block, "mimeType", "image/?")
            parts.append(f"[image content omitted, mime={mime}]")
        else:
            try:
                parts.append(json.dumps(block, default=str))
            except Exception:
                parts.append(str(block))
    out = "\n".join(parts).strip() or ("[empty result]" if not is_error else "[error]")
    if is_error:
        out = "[MCP error]\n" + out
    return out


def _make_caller(handle: MCPServerHandle, tool_name: str):
    """Build an async (args, cwd) -> str callable for register_tool."""

    async def call(args: dict, cwd: str) -> str:
        return await handle.call(tool_name, args)

    return call


class MCPManager:
    """Singleton coordinator for MCP servers tied to the active agent."""

    def __init__(self):
        self._active_agent: Optional[str] = None
        self._handles: Dict[str, MCPServerHandle] = {}
        self._registered_tool_names: Set[str] = set()
        self._lock = asyncio.Lock()

    async def activate_agent(self, agent_name: str, force: bool = False) -> None:
        """Tear down current servers, start the new agent's servers.

        Pass force=True to reload even if the same agent is already active.
        Failures starting individual servers are logged but don't abort the
        rest — the WebSocket conversation must keep working even if some
        MCP servers crash on launch.
        """
        async with self._lock:
            if not force and self._active_agent == agent_name and self._handles:
                return
            await self._deactivate_locked()

            servers = agents.read_mcp_servers(agent_name)
            self._active_agent = agent_name
            if not servers:
                return

            for srv_name, config in servers.items():
                try:
                    handle = MCPServerHandle(srv_name, config, agent_name=agent_name)
                    await handle.start()
                    self._handles[srv_name] = handle
                    self._register_tools_for(handle)
                except Exception as e:
                    logger.error(
                        f"Failed to start MCP server '{srv_name}' for agent '{agent_name}': {e}"
                    )

    async def deactivate(self) -> None:
        async with self._lock:
            await self._deactivate_locked()

    async def _deactivate_locked(self) -> None:
        for tool_name in list(self._registered_tool_names):
            unregister_tool(tool_name)
        self._registered_tool_names.clear()
        for handle in list(self._handles.values()):
            try:
                await handle.stop()
            except Exception as e:
                logger.error(f"Error stopping MCP server '{handle.name}': {e}")
        self._handles.clear()
        self._active_agent = None

    def _register_tools_for(self, handle: MCPServerHandle) -> None:
        for tool in handle.tools:
            tool_name = getattr(tool, "name", None)
            if not tool_name:
                continue
            qualified = f"mcp__{handle.name}__{tool_name}"
            description = getattr(tool, "description", "") or ""
            input_schema = getattr(tool, "inputSchema", None) or {
                "type": "object",
                "properties": {},
            }
            register_tool(ToolDef(
                name=qualified,
                schema={
                    "name": qualified,
                    "description": f"[MCP:{handle.name}] {description}",
                    "parameters": input_schema,
                },
                func=_make_caller(handle, tool_name),
                # MCP tools require user confirmation by default — the protocol
                # has no standard way to mark a tool as read-only and we can't
                # know whether a third-party server is safe.
                read_only=False,
            ))
            self._registered_tool_names.add(qualified)

    def loaded_servers(self) -> Dict[str, int]:
        return {name: len(h.tools) for name, h in self._handles.items()}


mcp_manager = MCPManager()
