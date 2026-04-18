"""
Tool definitions and execution for the agent.

Provides file operations, search, and command execution tools
in OpenAI function calling format.

Uses a registry pattern: each tool is a ToolDef with metadata
(read_only, browser_tool) used for auto-approval and permissions.
"""

import os
import re
import json
import glob as glob_mod
import asyncio
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Awaitable, Dict, Any

from ..log import ServiceLogger
from ..path_security import resolve_within_base

log = ServiceLogger("Tools")

# Maximum output sizes
MAX_OUTPUT = 10_000  # 10KB for bash
MAX_GREP_OUTPUT = 5_000
MAX_GLOB_RESULTS = 100
MAX_SEARCH_RESULTS = 5
MAX_FETCH_CHARS = 8000
MAX_TOOL_OUTPUT = 32_000

# ---------------------------------------------------------------------------
# Safe bash whitelist
# ---------------------------------------------------------------------------
SAFE_BASH_COMMANDS = {
    "ls", "pwd", "whoami", "date", "wc", "find",
    "which", "echo", "git status", "git log", "git diff", "git branch",
    "pip list", "python --version", "node --version", "npm --version",
    # NOTE: cat/head/tail intentionally excluded — the model should use
    # the native `read` tool instead for better UX (line numbers, offset/limit).
}


_SKILL_RUNNERS = {"node", "python", "python3", "py", "bash", "sh", "deno", "bun"}
_CD_SKILLS_PREFIX_RE = re.compile(
    r"^cd\s+(?:\./)?\.skills(?:/[\w\-./]+)?\s*&&\s*",
    re.IGNORECASE,
)
_MAX_BASH_TIMEOUT = 300


def is_safe_bash(command: str) -> bool:
    cmd = command.strip().split()[0] if command.strip() else ""
    full = command.strip()
    if cmd in SAFE_BASH_COMMANDS or full in SAFE_BASH_COMMANDS:
        return True
    # Trust agent-owned skill scripts: a runner (node/python/etc) operating on
    # something inside .skills/. The agent author wrote those skills themselves.
    # We require an actual runner to avoid auto-approving destructive ops like
    # `rm -rf .skills/...`. Two shapes are recognised:
    #   1. node ./.skills/<name>/cli.js ...
    #   2. cd ./.skills[/<name>] && node ...
    if ".skills" not in full:
        return False
    # Strip optional `cd <.skills path> && ` prefix
    m = _CD_SKILLS_PREFIX_RE.match(full)
    rest = full[m.end():] if m else full
    parts = rest.split(maxsplit=1)
    if not parts:
        return False
    return parts[0].lower() in _SKILL_RUNNERS


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------
@dataclass
class ToolDef:
    name: str
    schema: dict                               # OpenAI function schema (inner)
    func: Callable[..., Awaitable[str]]        # async function(args, cwd) -> str
    read_only: bool = True
    browser_tool: bool = False


TOOL_REGISTRY: Dict[str, ToolDef] = {}


def register_tool(tool_def: ToolDef):
    TOOL_REGISTRY[tool_def.name] = tool_def


def unregister_tool(name: str) -> bool:
    """Remove a tool from the registry. Returns True if the tool existed."""
    return TOOL_REGISTRY.pop(name, None) is not None


def get_tool_schemas() -> list:
    """Return tool schemas in OpenAI function calling format."""
    return [{"type": "function", "function": t.schema} for t in TOOL_REGISTRY.values()]


def is_dangerous(name: str, arguments: dict | None = None) -> bool:
    """Check if a tool call requires user confirmation."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return True  # unknown tools are dangerous
    if tool.browser_tool:
        return False
    if tool.read_only:
        return False
    # Special case: bash with safe command
    if name == "bash" and arguments:
        cmd = arguments.get("command", "")
        if is_safe_bash(cmd):
            return False
    return True


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------
def truncate_output(result: str, max_chars: int = MAX_TOOL_OUTPUT) -> str:
    if len(result) <= max_chars:
        return result
    first = max_chars // 2
    last = max_chars // 4
    return result[:first] + f"\n\n[... {len(result) - first - last} chars truncated ...]\n\n" + result[-last:]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _resolve(path: str, cwd: str) -> str:
    """Resolve a path relative to cwd and keep it sandboxed to that tree."""
    return str(resolve_within_base(Path(cwd).resolve(), path))


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
async def execute_tool(name: str, arguments: dict, cwd: str = ".") -> str:
    """Execute a tool by name and return the (truncated) result string."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return f"Unknown tool: {name}"
    try:
        result = await tool.func(arguments, cwd)
    except Exception as e:
        log.error(f"Tool {name} failed", e)
        result = f"Error: {e}"
    return truncate_output(result)


# ===========================================================================
# Tool implementations
# ===========================================================================

_BINARY_EXTS = {
    "jpg", "jpeg", "png", "gif", "webp", "bmp", "ico", "tiff", "svg",
    "mp3", "wav", "ogg", "flac", "m4a", "mp4", "mov", "avi", "webm", "mkv",
    "pdf", "zip", "tar", "gz", "7z", "rar", "exe", "dll", "so", "dylib",
    "bin", "dat", "db", "sqlite", "ttf", "otf", "woff", "woff2",
}


async def _tool_read(args: dict, cwd: str) -> str:
    path = _resolve(args["path"], cwd)
    offset = max(1, args.get("offset", 1))
    limit = args.get("limit", 200)

    if not os.path.isfile(path):
        return f"Error: File not found: {path}"

    ext = path.rsplit(".", 1)[-1].lower() if "." in os.path.basename(path) else ""
    if ext in _BINARY_EXTS:
        size = os.path.getsize(path)
        kind = "image" if ext in {"jpg","jpeg","png","gif","webp","bmp","ico","tiff","svg"} else \
               "audio" if ext in {"mp3","wav","ogg","flac","m4a"} else \
               "video" if ext in {"mp4","mov","avi","webm","mkv"} else "binary"
        if kind == "image":
            return (
                f"[{os.path.basename(path)} is an image ({size} bytes). "
                f"It has been auto-attached to the conversation for you to view if you have vision capability — "
                f"describe what you see directly. Do NOT say you cannot see images.]"
            )
        return (
            f"Error: '{os.path.basename(path)}' is a {kind} file ({size} bytes). "
            f"The read tool only handles text files. Reference it by path when answering."
        )

    # Sniff the first chunk for binary content (NUL byte) to catch unknown extensions.
    try:
        with open(path, "rb") as fb:
            head = fb.read(4096)
        if b"\x00" in head:
            size = os.path.getsize(path)
            return (
                f"Error: '{os.path.basename(path)}' appears to be binary ({size} bytes). "
                f"The read tool only handles text files."
            )
    except Exception:
        log.debug(f"Could not inspect file header for binary detection: {path}")

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
    """Edit a file by exact string match OR by line range.

    Two modes:
    1. old_string/new_string: replaces old_string with new_string (must be unique)
    2. line_start/line_end/new_string: replaces lines start..end (1-based, inclusive)
       with new_string. Use when exact string matching fails.
    """
    if "path" not in args:
        return "Error: 'path' parameter is required. Call edit with {path, new_string, old_string} or {path, new_string, line_start, line_end}."
    path = _resolve(args["path"], cwd)
    new_string = args.get("new_string", "")

    if not os.path.isfile(path):
        return f"Error: File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Mode 2: line-range based edit
    line_start = args.get("line_start")
    line_end = args.get("line_end")
    if line_start is not None and line_end is not None:
        try:
            ls = int(line_start)
            le = int(line_end)
        except (TypeError, ValueError):
            return "Error: line_start and line_end must be integers"
        lines = content.split("\n")
        if ls < 1 or le < ls or le > len(lines):
            return f"Error: invalid line range {ls}-{le} (file has {len(lines)} lines)"
        # Replace lines[ls-1..le-1] inclusive with new_string
        before = lines[:ls - 1]
        after = lines[le:]
        new_lines = new_string.split("\n") if new_string else []
        content = "\n".join(before + new_lines + after)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Edit applied: replaced lines {ls}-{le} ({le - ls + 1} line(s))"

    # Mode 1: exact string match
    old_string = args.get("old_string", "")
    if not old_string:
        return "Error: old_string (or line_start+line_end) is required"

    count = content.count(old_string)
    if count == 0:
        # Help the model self-correct: show a few candidate lines from the
        # file that share some overlap with the beginning of old_string.
        hint_key = old_string[:60].strip()
        lines = content.split("\n")
        close = []
        for i, line in enumerate(lines, 1):
            if hint_key[:20] in line or (len(hint_key) > 30 and hint_key[20:40] in line):
                close.append(f"  line {i}: {line[:120]}")
            if len(close) >= 3:
                break
        hint = ""
        if close:
            hint = "\nDid you mean one of these lines?\n" + "\n".join(close)
            hint += "\nTip: use line_start/line_end instead of old_string for precise edits."
        return f"Error: old_string not found in file.{hint}"
    if count > 1:
        return f"Error: old_string found {count} times (must be unique)"

    content = content.replace(old_string, new_string, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return "Edit applied successfully"


async def _tool_multi_edit(args: dict, cwd: str) -> str:
    path = _resolve(args["path"], cwd)
    edits = args.get("edits", [])

    # Robustness: if the model passed edits as a JSON string, clean up
    # common chat-template artifacts and try to parse it
    if isinstance(edits, str):
        # Strip leaked chat-template tokens (Gemma/Qwen produce <|"|> etc.)
        import re as _re
        cleaned = _re.sub(r'<\|[^>]*\|?>', '', edits)
        try:
            edits = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return (
                "Error: edits must be a JSON array of objects.\n"
                "Example: [{\"old_string\": \"old text\", \"new_string\": \"new text\"}]\n"
                "Or use line_start/line_end: [{\"line_start\": 8, \"line_end\": 8, \"new_string\": \"new line\"}]"
            )

    if not isinstance(edits, list):
        return "Error: edits must be a list of {old_string, new_string} or {line_start, line_end, new_string}"

    if not os.path.isfile(path):
        return f"Error: File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    applied = 0
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return f"Error in edit {i + 1}: each edit must be an object"

        # Support line-range mode (same as the edit tool)
        ls = edit.get("line_start")
        le = edit.get("line_end")
        new = edit.get("new_string", "")
        if ls is not None and le is not None:
            try:
                ls, le = int(ls), int(le)
            except (TypeError, ValueError):
                return f"Error in edit {i + 1}: line_start/line_end must be integers"
            lines = content.split("\n")
            if ls < 1 or le < ls or le > len(lines):
                return f"Error in edit {i + 1}: invalid line range {ls}-{le} (file has {len(lines)} lines)"
            before = lines[:ls - 1]
            after = lines[le:]
            new_lines = new.split("\n") if new else []
            content = "\n".join(before + new_lines + after)
            applied += 1
            continue

        old = edit.get("old_string", "")
        if not old:
            return f"Error in edit {i + 1}: old_string (or line_start+line_end) is required"
        count = content.count(old)
        if count == 0:
            return f"Error in edit {i + 1}: old_string not found"
        if count > 1:
            return f"Error in edit {i + 1}: old_string found {count} times"
        content = content.replace(old, new, 1)
        applied += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Applied {applied} edits successfully"


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


_BASH_EXE = None  # cached path to bash.exe on Windows


def _find_bash() -> Optional[str]:
    global _BASH_EXE
    if _BASH_EXE is not None:
        return _BASH_EXE or None
    import shutil as _shutil
    found = _shutil.which("bash")
    if not found:
        for cand in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            r"C:\Windows\System32\bash.exe",
        ):
            if os.path.isfile(cand):
                found = cand
                break
    _BASH_EXE = found or ""
    return _BASH_EXE or None


_RESERVED_PORT_PATTERNS = [
    re.compile(r"\b(?:-p|--port|PORT=|port=|port\s+|:)8080\b", re.IGNORECASE),
    re.compile(r"\bhttp\.server\s+8080\b", re.IGNORECASE),
    re.compile(r"\b(?:-p|--port|PORT=|port=|port\s+|:)3040\b", re.IGNORECASE),
    re.compile(r"\bhttp\.server\s+3040\b", re.IGNORECASE),
]


async def _tool_bash(args: dict, cwd: str) -> str:
    command = args.get("command")
    if not isinstance(command, str):
        return "Error: command must be a string"
    command = command.strip()
    if not command:
        return "Error: command cannot be empty"
    try:
        timeout = int(args.get("timeout", 30))
    except (TypeError, ValueError):
        return "Error: timeout must be an integer"
    if timeout < 1:
        return "Error: timeout must be greater than 0"
    if timeout > _MAX_BASH_TIMEOUT:
        timeout = _MAX_BASH_TIMEOUT

    for pat in _RESERVED_PORT_PATTERNS:
        if pat.search(command):
            return (
                "Error: This command binds to a reserved port (8080 = llama-server, "
                "3040 = AINow). Binding here will kill your own LLM connection. "
                "Use 3000, 5000, 8000, or 8888 instead."
            )

    try:
        # On Windows, create_subprocess_shell uses cmd.exe which mishandles
        # POSIX-style quotes and pipes (e.g. tree -I 'a|b'). Route through
        # bash.exe (Git Bash) when available so the model can write portable
        # bash commands including pipes, redirects, and command chains.
        if os.name == "nt":
            bash = _find_bash()
            if bash:
                proc = await asyncio.create_subprocess_exec(
                    bash, "-c", command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=cwd,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=cwd,
                )
        else:
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
            log.error("Failed to kill timed-out bash process")
        return f"Error: Command timed out after {timeout}s"


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _format_results(query: str, results: list, source: str) -> str:
    if not results:
        return f"No results found for: {query}"
    out = [f"Search results for '{query}' (via {source}):", ""]
    for r in results[:MAX_SEARCH_RESULTS]:
        title = (r.get("title") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        url = (r.get("url") or "").strip()
        if title:
            out.append(f"- {title}")
        if snippet:
            out.append(f"  {snippet}")
        if url:
            out.append(f"  {url}")
        out.append("")
    return "\n".join(out).rstrip()


async def _search_tavily(query: str) -> list:
    """Tavily AI search API (https://tavily.com). Free tier available."""
    import httpx
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": MAX_SEARCH_RESULTS},
        )
        r.raise_for_status()
        data = r.json()
    results = []
    for item in (data.get("results") or []):
        results.append({
            "title": item.get("title"),
            "snippet": item.get("content"),
            "url": item.get("url"),
        })
    return results


async def _search_serper(query: str) -> list:
    """Serper.dev (Google results). Free tier available."""
    import httpx
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not set")
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": MAX_SEARCH_RESULTS},
        )
        r.raise_for_status()
        data = r.json()
    results = []
    for item in (data.get("organic") or []):
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "url": item.get("link"),
        })
    return results


async def _search_ddg(query: str) -> list:
    """DuckDuckGo HTML scrape with cookie warmup + POST."""
    import httpx
    import re as _re
    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://duckduckgo.com/",
    }
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as c:
        # Cookie warmup
        await c.get("https://duckduckgo.com/")
        r = await c.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "kl": "us-en"},
        )
        if r.status_code != 200 or "result__a" not in r.text:
            raise RuntimeError(f"DDG returned {r.status_code} or no results")
        html = r.text
    blocks = _re.findall(
        r'<a rel="nofollow" class="result__a" href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'<a class="result__snippet"[^>]*>(.*?)</a>',
        html, _re.DOTALL,
    )
    results = []
    for url, title, snippet in blocks:
        title = _re.sub(r"<[^>]+>", "", title).strip()
        snippet = _re.sub(r"<[^>]+>", "", snippet).strip()
        # DDG wraps URLs in their own redirect — try to extract the real one
        if url.startswith("//"):
            url = "https:" + url
        results.append({"title": title, "snippet": snippet, "url": url})
    if not results:
        raise RuntimeError("DDG returned no parsable results")
    return results


async def _tool_web_search(args: dict, cwd: str = "") -> str:
    """Multi-backend web search.

    Backend priority (first one that works wins):
      1. Tavily Search API    — set TAVILY_API_KEY (free tier, no card)
      2. Serper.dev (Google)  — set SERPER_API_KEY (free tier, no card)
      3. DuckDuckGo HTML scrape (often blocked from cloud IPs)
    """
    query = args.get("query", "")
    if not query:
        return "Error: query is required"

    backends = []
    if os.getenv("TAVILY_API_KEY"):
        backends.append(("Tavily", _search_tavily))
    if os.getenv("SERPER_API_KEY"):
        backends.append(("Serper", _search_serper))
    backends.append(("DuckDuckGo", _search_ddg))

    errors = []
    for name, fn in backends:
        try:
            results = await fn(query)
            if results:
                return _format_results(query, results, name)
            errors.append(f"{name}: empty results")
        except Exception as e:
            errors.append(f"{name}: {e}")
            log.info(f"web_search backend '{name}' failed: {e}")

    return (
        f"Error: all web search backends failed for query '{query}'.\n"
        f"Tried: {', '.join(name for name, _ in backends)}.\n\n"
        "Hint: set TAVILY_API_KEY (free tier at tavily.com, no credit card) "
        "or SERPER_API_KEY (free tier at serper.dev) in your .env for reliable search. "
        "Or configure an MCP search server (Tavily/Exa/Firecrawl) "
        "via the MCP Servers UI on the active agent.\n\n"
        f"Details: {'; '.join(errors[:3])}"
    )


async def _tool_web_fetch(args: dict, cwd: str = "") -> str:
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


async def _noop_browser_tool(args: dict, cwd: str) -> str:
    """Placeholder for browser tools — actual dispatch handled by LLM service callback."""
    return ""


# ===========================================================================
# Register all tools
# ===========================================================================

def _register_all():
    register_tool(ToolDef(
        name="read",
        schema={
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
        func=_tool_read,
        read_only=True,
    ))

    register_tool(ToolDef(
        name="write",
        schema={
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
        func=_tool_write,
        read_only=False,
    ))

    register_tool(ToolDef(
        name="edit",
        schema={
            "name": "edit",
            "description": "Edit a file. Two modes: (1) old_string+new_string for exact string replacement, (2) line_start+line_end+new_string for line-range replacement (use line numbers from the read tool output). Line mode is more reliable when exact string matching fails.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_string": {"type": "string", "description": "Exact text to find (must be unique). Omit if using line_start/line_end."},
                    "new_string": {"type": "string", "description": "Replacement text"},
                    "line_start": {"type": "integer", "description": "First line to replace (1-based, inclusive). Use with line_end."},
                    "line_end": {"type": "integer", "description": "Last line to replace (1-based, inclusive). Use with line_start."},
                },
                "required": ["path", "new_string"],
            },
        },
        func=_tool_edit,
        read_only=False,
    ))

    register_tool(ToolDef(
        name="multi_edit",
        schema={
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
        func=_tool_multi_edit,
        read_only=False,
    ))

    register_tool(ToolDef(
        name="grep",
        schema={
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
        func=_tool_grep,
        read_only=True,
    ))

    register_tool(ToolDef(
        name="glob",
        schema={
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
        func=_tool_glob,
        read_only=True,
    ))

    register_tool(ToolDef(
        name="ls",
        schema={
            "name": "ls",
            "description": "List directory contents with d/ (directory) or f/ (file) prefix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Default: current dir"},
                },
            },
        },
        func=_tool_ls,
        read_only=True,
    ))

    register_tool(ToolDef(
        name="bash",
        schema={
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
        func=_tool_bash,
        read_only=False,  # dangerous by default; is_dangerous() checks safe list
    ))

    register_tool(ToolDef(
        name="list_devices",
        schema={
            "name": "list_devices",
            "description": "List available video input devices (cameras, screens) and their active status.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        func=_noop_browser_tool,
        browser_tool=True,
    ))

    register_tool(ToolDef(
        name="capture_frame",
        schema={
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
        func=_noop_browser_tool,
        browser_tool=True,
    ))

    register_tool(ToolDef(
        name="web_search",
        schema={
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
        func=_tool_web_search,
        read_only=True,
    ))

    register_tool(ToolDef(
        name="web_fetch",
        schema={
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
        func=_tool_web_fetch,
        read_only=True,
    ))


# Auto-register on import
_register_all()
