# AINow - Agentic Loop Improvements

Based on analysis of [nano-claude-code](https://github.com/SafeRL-Lab/nano-claude-code).

## Priority 1: Context Management (long conversations break)

- [ ] **Snip old tool results** — Truncate tool output from early turns (keep first 50% + last 25%, insert `[... N chars snipped ...]`). Fast, no LLM call needed.
- [ ] **Auto-compact** — When over 70% of context limit, use LLM to summarize old messages. Keep last 6 turns intact, replace older with summary.
- [ ] **Token estimation** — Simple heuristic: `total_chars / 3.5`. Track per-turn token usage.
- [ ] **Tool output cap** — Max 32KB per tool result at dispatcher level. Prevents one huge result from eating all context.

## Priority 2: Streaming Events (better UI feedback)

- [ ] **Event types** — Emit distinct events: `TextChunk`, `ToolStart`, `ToolEnd`, `TurnDone`, `ThinkingChunk`
- [ ] **Tool call preview** — Show tool name + args in UI before execution starts
- [ ] **Tool result feedback** — Show result/error in UI after execution
- [ ] **Spinner/status** — Show "thinking..." or "executing tool..." states in UI

## Priority 3: Tool Registry (pluggable tools)

- [ ] **ToolDef dataclass** — `name`, `schema`, `func`, `read_only`, `concurrent_safe`
- [ ] **register_tool()** — Register tools dynamically instead of hardcoding
- [ ] **Semantic permissions** — Auto-approve read-only tools (read, glob, grep, web_search, web_fetch). Ask for write/edit/bash.
- [ ] **Safe bash whitelist** — Auto-approve: `ls`, `git status`, `find`, `cat`, `pwd`. Ask for everything else.

## Priority 4: Dynamic System Prompt

- [ ] **Inject context** — Current date/time, working directory, platform, git branch
- [ ] **Project instructions** — Load CLAUDE.md or .ainow/instructions.md if exists
- [ ] **Available tools list** — Auto-generate from tool registry
- [ ] **Memory injection** — Load persistent memories into system prompt

## Priority 5: Memory System

- [ ] **File-based storage** — `~/.ainow/memory/*.md` (global) + `.ainow/memory/*.md` (project)
- [ ] **MEMORY.md index** — Auto-rebuilt, truncated to 200 lines
- [ ] **Memory types** — user, feedback, project, reference (with YAML frontmatter)
- [ ] **Tools** — `memory_save`, `memory_delete`, `memory_search`, `memory_list`
- [ ] **Inject into prompt** — Load relevant memories at conversation start

## Priority 6: Multi-Provider Support

- [ ] **Neutral message format** — Internal format independent of any provider:
  ```
  {role: "user", content: "text"}
  {role: "assistant", content: "text", tool_calls: [...]}
  {role: "tool", tool_call_id: "...", name: "...", content: "..."}
  ```
- [ ] **Provider adapters** — Convert to/from OpenAI, Anthropic, Google formats
- [ ] **Model picker in UI** — Switch models without restarting

## Priority 7: Error Recovery

- [ ] **Tool error wrapping** — All tool execution in try/except, return errors as strings
- [ ] **Provider retry** — Retry on 429 (rate limit) and 5xx with exponential backoff
- [ ] **No infinite loops** — Max tool call iterations per turn (e.g. 10)
- [ ] **Graceful degradation** — If tool fails, tell LLM it failed and let it adapt

## Priority 8: Sub-Agents

- [ ] **Agent tool** — LLM can spawn sub-agents for complex tasks
- [ ] **Agent types** — researcher (web), coder (files), planner (decompose tasks)
- [ ] **Background agents** — Run in background, check results later
- [ ] **Max depth** — Prevent infinite agent recursion

## Priority 9: Voice-Specific Features

- [ ] **Timer/reminder tool** — "Remind me in 5 minutes" via background thread
- [ ] **Proactive monitoring** — "Check X every N minutes" pattern
- [ ] **Voice command shortcuts** — "stop", "cancel", "repeat", "louder", "slower"
- [ ] **Conversation bookmarks** — "Remember this point" for voice navigation

## Quick Wins (can do now)

- [ ] Track token count per turn (already partially done in UI)
- [ ] Max tool iterations guard (prevent infinite tool loops)
- [ ] Truncate large tool results before feeding back to LLM
- [ ] Auto-approve read-only tools (no confirmation for read/grep/glob/web_search)
