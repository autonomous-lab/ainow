"""Tolerant parser for small-model tool-call JSON.

Small LLMs (2B–9B) regularly emit malformed JSON in the `function.arguments`
field of a tool call: literal newlines inside strings, trailing commas, single
quotes, unquoted keys, missing closing braces. Failing fast on those means
the turn silently loses the tool call.

This parser applies progressive repairs before giving up. Inspired by
little-coder's `local/output_parser.py`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_TRAILING_COMMA_OBJ = re.compile(r",\s*}")
_TRAILING_COMMA_ARR = re.compile(r",\s*]")
_UNQUOTED_KEY = re.compile(r"(?<=[{,\s])([A-Za-z_][A-Za-z0-9_]*)\s*:")
_BARE_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _escape_newlines_in_strings(text: str) -> str:
    """Replace literal newlines / tabs inside quoted strings with escape seqs.

    We track whether we're inside a double-quoted string, respecting backslash
    escapes. Single-quoted strings are NOT treated as JSON strings here (those
    get normalized elsewhere).
    """
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    text = _TRAILING_COMMA_OBJ.sub("}", text)
    text = _TRAILING_COMMA_ARR.sub("]", text)
    return text


def _normalize_quotes(text: str) -> str:
    """If the text uses single quotes and no double quotes, swap them.

    We avoid blind substitution when both kinds are present (apostrophes in
    English text would break things).
    """
    if '"' in text:
        return text
    if "'" not in text:
        return text
    # Safe swap: apostrophes inside single-quoted values become broken, but that's
    # a small regression next to the wholesale failure of single-quoted JSON.
    return text.replace("'", '"')


def _quote_unquoted_keys(text: str) -> str:
    """Wrap bare word keys as `"key":`. Skip if keys already look quoted."""
    if '": ' in text or '":"' in text:
        return text
    return _UNQUOTED_KEY.sub(r'"\1":', text)


def _balance_braces(text: str) -> str:
    """Append missing closing braces/brackets so json.loads can proceed.

    Count unmatched `{` and `[` only (we don't try to fix mismatched closers).
    """
    open_curly = text.count("{") - text.count("}")
    open_square = text.count("[") - text.count("]")
    if open_curly > 0:
        text = text + ("}" * open_curly)
    if open_square > 0:
        text = text + ("]" * open_square)
    return text


def _try_loads(text: str) -> Optional[dict]:
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def parse_tool_arguments(raw: str) -> dict:
    """Parse a tool-call arguments string.

    Returns an empty dict on total failure (caller can log and skip). Applies
    progressive repairs between attempts. Prefer this over bare `json.loads`.
    """
    if not raw:
        return {}
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else {}

    text = raw.strip()
    if not text:
        return {}

    # Fast path
    result = _try_loads(text)
    if result is not None:
        return result

    # Progressive repairs
    for fix in (
        _escape_newlines_in_strings,
        _strip_trailing_commas,
        _normalize_quotes,
        _quote_unquoted_keys,
        _balance_braces,
    ):
        text = fix(text)
        result = _try_loads(text)
        if result is not None:
            return result

    # Fallback: extract the first bare `{...}` and try parsing that.
    m = _BARE_OBJECT.search(raw)
    if m:
        candidate = m.group(0)
        for fix in (_strip_trailing_commas, _normalize_quotes, _quote_unquoted_keys, _balance_braces):
            candidate = fix(candidate)
            result = _try_loads(candidate)
            if result is not None:
                return result

    return {}


def coerce_value(value: Any, target_type: str) -> Any:
    """Coerce a string value to int/float/bool when the schema expects it.

    Used after parse_tool_arguments: small models often stringify booleans/ints.
    """
    if target_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "1", "yes", "on"):
                return True
            if s in ("false", "0", "no", "off"):
                return False
        return bool(value)
    if target_type == "integer":
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if target_type == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


__all__ = ["parse_tool_arguments", "coerce_value"]
