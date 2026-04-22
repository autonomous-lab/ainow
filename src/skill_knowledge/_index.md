# Skill Knowledge Packs

Each `.md` file in this directory is a compact guidance pack injected into the
system prompt when its trigger matches the user message or the active tool call.

## Frontmatter format

Each file begins with YAML-ish frontmatter:

```
---
name: short_name
triggers: [keyword, "exact phrase", regex:/pattern/i]
tools: [tool_name, mcp__server__tool]      # optional
max_chars: 2000                             # optional, default 3000
---
```

Triggers match if **any** of them fires. A `triggers:` entry prefixed with
`regex:` is compiled as a regex pattern; other entries are substring matches
(case-insensitive).

Tools list is an alternative activation path: the pack is attached for the
**next turn** after any of the listed tools is invoked.

Keep each pack under 500 tokens. Prefer 3-5 bullet points of *specific,
actionable* guidance over long prose.
