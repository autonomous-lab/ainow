"""Skill knowledge packs: compact per-task guidance injected into the system prompt.

Each .md file under this directory has YAML-ish frontmatter declaring its
triggers (substring or regex matches on the user message) and/or tools (tool
names that activate it on the next turn). Matching packs are appended to the
system prompt so the model has focused, actionable guidance when it needs it —
without paying the token cost when it doesn't.

Inspired by little-coder's skill/knowledge pattern.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set


@dataclass
class SkillPack:
    name: str
    body: str
    triggers: List[str] = field(default_factory=list)  # raw substring
    trigger_regexes: List[re.Pattern] = field(default_factory=list)
    tools: Set[str] = field(default_factory=set)
    max_chars: int = 3000
    path: Optional[Path] = None

    def matches_text(self, text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        for t in self.triggers:
            if t.lower() in low:
                return True
        for rx in self.trigger_regexes:
            if rx.search(text):
                return True
        return False

    def matches_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Tolerant YAML-ish frontmatter parser. Only supports the keys we use."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end < 0:
        return {}, raw
    header = raw[3:end].strip()
    body = raw[end + 4 :].lstrip("\n")
    meta: dict = {}
    for line in header.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [x.strip().strip('"').strip("'") for x in val[1:-1].split(",")]
            meta[key] = [x for x in items if x]
        else:
            meta[key] = val.strip('"').strip("'")
    return meta, body


def _load_one(path: Path) -> Optional[SkillPack]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, body = _parse_frontmatter(raw)
    if not body.strip():
        return None
    name = str(meta.get("name") or path.stem)
    triggers: List[str] = []
    trigger_regexes: List[re.Pattern] = []
    for t in meta.get("triggers", []) or []:
        if isinstance(t, str) and t.startswith("regex:"):
            pat = t[len("regex:") :]
            # Accept /pattern/flags or plain
            flags = 0
            if pat.startswith("/"):
                close = pat.rfind("/")
                if close > 0:
                    flag_part = pat[close + 1 :]
                    pat = pat[1:close]
                    if "i" in flag_part:
                        flags |= re.IGNORECASE
                    if "m" in flag_part:
                        flags |= re.MULTILINE
                    if "s" in flag_part:
                        flags |= re.DOTALL
            try:
                trigger_regexes.append(re.compile(pat, flags))
            except re.error:
                continue
        elif isinstance(t, str) and t:
            triggers.append(t)
    tools = set(t for t in (meta.get("tools", []) or []) if isinstance(t, str))
    try:
        max_chars = int(meta.get("max_chars", 3000))
    except (TypeError, ValueError):
        max_chars = 3000
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars] + "\n..."
    return SkillPack(
        name=name,
        body=body,
        triggers=triggers,
        trigger_regexes=trigger_regexes,
        tools=tools,
        max_chars=max_chars,
        path=path,
    )


_PACKS_CACHE: Optional[List[SkillPack]] = None
_PACKS_DIR = Path(__file__).parent


def load_packs(refresh: bool = False) -> List[SkillPack]:
    """Load all skill packs from this directory. Cached until `refresh=True`."""
    global _PACKS_CACHE
    if _PACKS_CACHE is not None and not refresh:
        return _PACKS_CACHE
    packs: List[SkillPack] = []
    for p in sorted(_PACKS_DIR.glob("*.md")):
        if p.name.startswith("_"):
            continue
        pack = _load_one(p)
        if pack:
            packs.append(pack)
    _PACKS_CACHE = packs
    return packs


def select_packs(
    user_text: str = "",
    last_tools: Optional[Iterable[str]] = None,
    last_failed_tool: Optional[str] = None,
    max_packs: int = 3,
    token_budget: int = 500,
    refresh: bool = False,
) -> List[SkillPack]:
    """Pick packs with a priority-ranked search.

    Priority (highest first, inspired by little-coder):
      1. Error recovery — a pack whose `tools:` includes the last-failed tool.
      2. Recency — packs tied to tools used in the past 2 turns.
      3. Intent — substring / regex triggers on the current user message.

    A global `token_budget` is enforced (chars ≈ tokens × 3.5) so a single
    greedy pack can't crowd out the whole system prompt.
    """
    packs = load_packs(refresh=refresh)
    if not packs:
        return []

    selected: List[SkillPack] = []
    used_budget_chars = 0
    byte_budget = int(token_budget * 3.5) if token_budget > 0 else 0

    def _try_add(p: SkillPack) -> bool:
        nonlocal used_budget_chars
        if p in selected:
            return False
        if len(selected) >= max_packs:
            return False
        if byte_budget and used_budget_chars + len(p.body) > byte_budget:
            return False
        selected.append(p)
        used_budget_chars += len(p.body)
        return True

    # 1) Error recovery — highest priority
    if last_failed_tool:
        for p in packs:
            if p.matches_tool(last_failed_tool):
                _try_add(p)

    # 2) Recency
    last_tool_set: Set[str] = set(last_tools or [])
    if last_tool_set:
        for p in packs:
            if any(p.matches_tool(t) for t in last_tool_set):
                _try_add(p)

    # 3) Intent (user-message triggers)
    if user_text:
        for p in packs:
            if p.matches_text(user_text):
                _try_add(p)

    return selected


def render_packs(packs: Iterable[SkillPack]) -> str:
    """Render selected packs into a system-prompt section."""
    out: List[str] = []
    for p in packs:
        out.append(p.body)
    return "\n\n---\n\n".join(out)


__all__ = ["SkillPack", "load_packs", "select_packs", "render_packs"]
