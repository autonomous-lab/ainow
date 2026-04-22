---
name: file_edits
triggers: ["edit a file", "modify the file", "change the code", "refactor"]
tools: [write, edit, multi_edit]
max_chars: 1500
---

## File editing — do it right

- **Read before you edit**: always `read` the target file first to see the exact current content. Your memory of the file from 2 turns ago is probably stale.
- **`write` refuses to overwrite existing files**. If the file exists, use `edit` (single string) or `multi_edit` (batch). Use `write` with `overwrite: true` ONLY when the intent is a full replacement (rare).
- **`edit` needs a UNIQUE `old_string`**. If the snippet appears multiple times, include more surrounding context until it's unique, or use `replace_all`.
- **Indentation matters**: when copying `old_string`, preserve the exact leading whitespace — mixing tabs and spaces makes the match fail silently.
- After an edit, **read the file again** to confirm the change landed correctly. Don't assume success from the tool's "Edit succeeded" message.
- For line-number edits: `line_start` / `line_end` are 1-based and inclusive. The range is replaced in full by `new_string`.
- Don't append trailing blank lines or strip the final newline unintentionally.
