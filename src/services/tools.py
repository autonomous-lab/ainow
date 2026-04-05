"""
Tool definitions and execution for the agent.

Provides file operations, search, and command execution tools
in OpenAI function calling format.
"""

import os
import glob as glob_mod
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

from ..log import ServiceLogger

log = ServiceLogger("Tools")

# Maximum output sizes
MAX_OUTPUT = 10_000  # 10KB for bash
MAX_GREP_OUTPUT = 5_000
MAX_GLOB_RESULTS = 100

DANGEROUS_TOOLS = {"write", "edit", "multi_edit", "bash"}
BROWSER_TOOLS = {"list_devices", "capture_frame"}
MAX_SEARCH_RESULTS = 5
MAX_FETCH_CHARS = 8000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read file contents. Returns numbered lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "offset": {"type": "integer", "description": "Start line (1-based). Default: 1"},
                    "limit": {"type": "integer", "description": "Max lines to return. Default: 200"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace an exact string in a file. old_string must be unique in the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_string": {"type": "string", "description": "Exact text to find (must be unique)"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_edit",
            "description": "Apply multiple sequential edits to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                            },
                            "required": ["old_string", "new_string"],
                        },
                        "description": "List of {old_string, new_string} edits to apply in order",
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents using ripgrep (rg). Returns matching lines with file and line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "File or directory to search in. Default: current dir"},
                    "glob": {"type": "string", "description": "Glob filter for files, e.g. '*.py'"},
                    "case_insensitive": {"type": "boolean", "description": "Case insensitive search. Default: false"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern. Returns up to 100 file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'"},
                    "path": {"type": "string", "description": "Base directory. Default: current dir"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List directory contents with d/ (directory) or f/ (file) prefix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Default: current dir"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds. Default: 30"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_devices",
            "description": "List available video input devices (cameras, screens) and their active status.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_frame",
            "description": "Take a photo using the user's webcam or capture their screen. Returns the image for you to see and analyze. Auto-starts the camera/screen if needed. Use source='webcam' to photograph the user, source='screen' to screenshot their display.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["webcam", "screen"],
                        "description": "Video source to capture from: 'webcam' or 'screen'",
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use when the user asks about recent events, news, prices, weather, or anything that requires up-to-date information. Do NOT use for general knowledge questions like math, definitions, or well-known facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch the content of a web page given its URL. Use after web_search to read a specific page, or when the user provides a URL directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
]


def _resolve(path: str, cwd: str) -> str:
    """Resolve a path relative to cwd."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return str(p.resolve())


async def execute_tool(name: str, arguments: dict, cwd: str = ".") -> str:
    """Execute a tool by name and return the result string."""
    try:
        if name == "read":
            return await _tool_read(arguments, cwd)
        elif name == "write":
            return await _tool_write(arguments, cwd)
        elif name == "edit":
            return await _tool_edit(arguments, cwd)
        elif name == "multi_edit":
            return await _tool_multi_edit(arguments, cwd)
        elif name == "grep":
            return await _tool_grep(arguments, cwd)
        elif name == "glob":
            return await _tool_glob(arguments, cwd)
        elif name == "ls":
            return await _tool_ls(arguments, cwd)
        elif name == "bash":
            return await _tool_bash(arguments, cwd)
        elif name == "web_search":
            return await _tool_web_search(arguments)
        elif name == "web_fetch":
            return await _tool_web_fetch(arguments)
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        log.error(f"Tool {name} failed", e)
        return f"Error: {e}"


async def _tool_read(args: dict, cwd: str) -> str:
    path = _resolve(args["path"], cwd)
    offset = max(1, args.get("offset", 1))
    limit = args.get("limit", 200)

    if not os.path.isfile(path):
        return f"Error: File not found: {path}"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    selected = lines[offset - 1 : offset - 1 + limit]
    numbered = []
    for i, line in enumerate(selected, start=offset):
        numbered.append(f"{i:>6}\t{line.rstrip()}")
    return "\n".join(numbered) if numbered else "(empty file)"


async def _tool_write(args: dict, cwd: str) -> str:
    path = _resolve(args["path"], cwd)
    content = args["content"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"


async def _tool_edit(args: dict, cwd: str) -> str:
    path = _resolve(args["path"], cwd)
    old_string = args["old_string"]
    new_string = args["new_string"]

    if not os.path.isfile(path):
        return f"Error: File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    count = content.count(old_string)
    if count == 0:
        return "Error: old_string not found in file"
    if count > 1:
        return f"Error: old_string found {count} times (must be unique)"

    content = content.replace(old_string, new_string, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return "Edit applied successfully"


async def _tool_multi_edit(args: dict, cwd: str) -> str:
    path = _resolve(args["path"], cwd)
    edits = args.get("edits", [])

    if not os.path.isfile(path):
        return f"Error: File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    for i, edit in enumerate(edits):
        old = edit["old_string"]
        new = edit["new_string"]
        count = content.count(old)
        if count == 0:
            return f"Error in edit {i + 1}: old_string not found"
        if count > 1:
            return f"Error in edit {i + 1}: old_string found {count} times"
        content = content.replace(old, new, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Applied {len(edits)} edits successfully"


async def _tool_grep(args: dict, cwd: str) -> str:
    pattern = args["pattern"]
    path = args.get("path", cwd) or cwd
    path = _resolve(path, cwd)
    file_glob = args.get("glob")
    case_insensitive = args.get("case_insensitive", False)

    cmd = ["rg", "--no-heading", "-n", "--max-count", "50"]
    if case_insensitive:
        cmd.append("-i")
    if file_glob:
        cmd.extend(["--glob", file_glob])
    cmd.extend([pattern, path])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > MAX_GREP_OUTPUT:
            output = output[:MAX_GREP_OUTPUT] + "\n... (truncated)"
        return output if output.strip() else "No matches found"
    except FileNotFoundError:
        # rg not available, fall back to grep
        cmd[0] = "grep"
        cmd = ["grep", "-rn"]
        if case_insensitive:
            cmd.append("-i")
        cmd.extend([pattern, path])
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > MAX_GREP_OUTPUT:
            output = output[:MAX_GREP_OUTPUT] + "\n... (truncated)"
        return output if output.strip() else "No matches found"
    except asyncio.TimeoutError:
        return "Error: grep timed out after 15s"


async def _tool_glob(args: dict, cwd: str) -> str:
    pattern = args["pattern"]
    base = args.get("path", cwd) or cwd
    base = _resolve(base, cwd)

    full_pattern = os.path.join(base, pattern)
    matches = sorted(glob_mod.glob(full_pattern, recursive=True))[:MAX_GLOB_RESULTS]

    if not matches:
        return "No files matched"

    # Show paths relative to base
    result = []
    for m in matches:
        try:
            rel = os.path.relpath(m, base)
        except ValueError:
            rel = m
        result.append(rel)
    return "\n".join(result)


async def _tool_ls(args: dict, cwd: str) -> str:
    path = args.get("path", cwd) or cwd
    path = _resolve(path, cwd)

    if not os.path.isdir(path):
        return f"Error: Not a directory: {path}"

    entries = sorted(os.listdir(path))
    result = []
    for entry in entries:
        full = os.path.join(path, entry)
        prefix = "d/" if os.path.isdir(full) else "f/"
        result.append(prefix + entry)
    return "\n".join(result) if result else "(empty directory)"


async def _tool_bash(args: dict, cwd: str) -> str:
    command = args["command"]
    timeout = args.get("timeout", 30)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace")
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + "\n... (truncated)"
        exit_info = f"\n[exit code: {proc.returncode}]" if proc.returncode != 0 else ""
        return (output + exit_info) if output.strip() or exit_info else "(no output)"
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"Error: Command timed out after {timeout}s"


async def _tool_web_search(args: dict) -> str:
    """Search the web using DuckDuckGo HTML."""
    import httpx
    import re

    query = args.get("query", "")
    if not query:
        return "Error: query is required"

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            html = resp.text

        # Parse results from DDG HTML
        results = []
        # Each result is in a <div class="result ...">
        blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        )
        for url, title, snippet in blocks[:MAX_SEARCH_RESULTS]:
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            results.append(f"- {title}\n  {snippet}\n  {url}")

        if not results:
            return f"No results found for: {query}"

        return f"Search results for '{query}':\n\n" + "\n\n".join(results)

    except Exception as e:
        log.error("Web search failed", e)
        return f"Error: Search failed — {e}"


async def _tool_web_fetch(args: dict) -> str:
    """Fetch a web page and return its text content."""
    import httpx
    import re

    url = args.get("url", "")
    if not url:
        return "Error: url is required"

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            html = resp.text

        # Strip scripts, styles, and tags to get readable text
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return f"No readable content found at {url}"

        if len(text) > MAX_FETCH_CHARS:
            text = text[:MAX_FETCH_CHARS] + "\n\n[...truncated]"

        return f"Content from {url}:\n\n{text}"

    except Exception as e:
        log.error("Web fetch failed", e)
        return f"Error: Fetch failed — {e}"
