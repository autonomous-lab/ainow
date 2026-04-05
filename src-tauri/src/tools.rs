//! Tool definitions and execution for the agent.
//!
//! Provides file operations, search, web tools, and command execution
//! in OpenAI function calling format.

use anyhow::Result;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use tokio::process::Command;

/// Maximum output sizes.
const MAX_OUTPUT: usize = 10_000;
const MAX_GREP_OUTPUT: usize = 5_000;
const MAX_GLOB_RESULTS: usize = 100;
const MAX_SEARCH_RESULTS: usize = 5;
const MAX_FETCH_CHARS: usize = 8_000;

/// Tools that require user confirmation before execution.
pub const DANGEROUS_TOOLS: &[&str] = &["write", "edit", "multi_edit", "bash"];

/// Tools that are dispatched to the browser, not executed server-side.
pub const BROWSER_TOOLS: &[&str] = &["list_devices", "capture_frame"];

/// Returns the OpenAI function calling tool definitions.
pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read file contents. Returns numbered lines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                        "offset": {"type": "integer", "description": "Start line (1-based). Default: 1"},
                        "limit": {"type": "integer", "description": "Max lines to return. Default: 200"}
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write",
                "description": "Create or overwrite a file with the given content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {"type": "string", "description": "File content to write"}
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Replace an exact string in a file. old_string must be unique in the file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to edit"},
                        "old_string": {"type": "string", "description": "Exact text to find (must be unique)"},
                        "new_string": {"type": "string", "description": "Replacement text"}
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            }
        }),
        json!({
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
                                    "new_string": {"type": "string"}
                                },
                                "required": ["old_string", "new_string"]
                            },
                            "description": "List of {old_string, new_string} edits to apply in order"
                        }
                    },
                    "required": ["path", "edits"]
                }
            }
        }),
        json!({
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
                        "case_insensitive": {"type": "boolean", "description": "Case insensitive search. Default: false"}
                    },
                    "required": ["pattern"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files matching a glob pattern. Returns up to 100 file paths.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'"},
                        "path": {"type": "string", "description": "Base directory. Default: current dir"}
                    },
                    "required": ["pattern"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "ls",
                "description": "List directory contents with d/ (directory) or f/ (file) prefix.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path. Default: current dir"}
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command and return stdout+stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds. Default: 30"}
                    },
                    "required": ["command"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_devices",
                "description": "List available video input devices (cameras, screens) and their active status.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }),
        json!({
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
                            "description": "Video source to capture from: 'webcam' or 'screen'"
                        }
                    },
                    "required": ["source"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information. Use when the user asks about recent events, news, prices, weather, or anything that requires up-to-date information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch the content of a web page given its URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to fetch"}
                    },
                    "required": ["url"]
                }
            }
        }),
    ]
}

/// Resolve a path relative to cwd.
fn resolve_path(path: &str, cwd: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        Path::new(cwd).join(p)
    }
}

/// Execute a tool by name and return the result string.
pub async fn execute_tool(name: &str, args: &Value, cwd: &str) -> String {
    match execute_tool_inner(name, args, cwd).await {
        Ok(result) => result,
        Err(e) => format!("Error: {e}"),
    }
}

async fn execute_tool_inner(name: &str, args: &Value, cwd: &str) -> Result<String> {
    match name {
        "read" => tool_read(args, cwd).await,
        "write" => tool_write(args, cwd).await,
        "edit" => tool_edit(args, cwd).await,
        "multi_edit" => tool_multi_edit(args, cwd).await,
        "grep" => tool_grep(args, cwd).await,
        "glob" => tool_glob(args, cwd).await,
        "ls" => tool_ls(args, cwd).await,
        "bash" => tool_bash(args, cwd).await,
        "web_search" => tool_web_search(args).await,
        "web_fetch" => tool_web_fetch(args).await,
        _ => Ok(format!("Unknown tool: {name}")),
    }
}

async fn tool_read(args: &Value, cwd: &str) -> Result<String> {
    let path_str = args["path"].as_str().unwrap_or("");
    let path = resolve_path(path_str, cwd);
    let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(200) as usize;

    if !path.is_file() {
        return Ok(format!("Error: File not found: {}", path.display()));
    }

    let content = tokio::fs::read_to_string(&path).await?;
    let lines: Vec<&str> = content.lines().collect();
    let start = (offset - 1).min(lines.len());
    let end = (start + limit).min(lines.len());

    if start >= lines.len() {
        return Ok("(empty file)".to_string());
    }

    let numbered: Vec<String> = lines[start..end]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{}", start + i + 1, line))
        .collect();

    if numbered.is_empty() {
        Ok("(empty file)".to_string())
    } else {
        Ok(numbered.join("\n"))
    }
}

async fn tool_write(args: &Value, cwd: &str) -> Result<String> {
    let path_str = args["path"].as_str().unwrap_or("");
    let content = args["content"].as_str().unwrap_or("");
    let path = resolve_path(path_str, cwd);

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(&path, content).await?;
    Ok(format!("Wrote {} bytes to {}", content.len(), path.display()))
}

async fn tool_edit(args: &Value, cwd: &str) -> Result<String> {
    let path_str = args["path"].as_str().unwrap_or("");
    let old_string = args["old_string"].as_str().unwrap_or("");
    let new_string = args["new_string"].as_str().unwrap_or("");
    let path = resolve_path(path_str, cwd);

    if !path.is_file() {
        return Ok(format!("Error: File not found: {}", path.display()));
    }

    let content = tokio::fs::read_to_string(&path).await?;
    let count = content.matches(old_string).count();

    if count == 0 {
        return Ok("Error: old_string not found in file".to_string());
    }
    if count > 1 {
        return Ok(format!("Error: old_string found {count} times (must be unique)"));
    }

    let new_content = content.replacen(old_string, new_string, 1);
    tokio::fs::write(&path, new_content).await?;
    Ok("Edit applied successfully".to_string())
}

async fn tool_multi_edit(args: &Value, cwd: &str) -> Result<String> {
    let path_str = args["path"].as_str().unwrap_or("");
    let edits = args.get("edits").and_then(|v| v.as_array());
    let path = resolve_path(path_str, cwd);

    if !path.is_file() {
        return Ok(format!("Error: File not found: {}", path.display()));
    }

    let edits = match edits {
        Some(e) => e,
        None => return Ok("Error: edits array required".to_string()),
    };

    let mut content = tokio::fs::read_to_string(&path).await?;

    for (i, edit) in edits.iter().enumerate() {
        let old = edit["old_string"].as_str().unwrap_or("");
        let new = edit["new_string"].as_str().unwrap_or("");
        let count = content.matches(old).count();
        if count == 0 {
            return Ok(format!("Error in edit {}: old_string not found", i + 1));
        }
        if count > 1 {
            return Ok(format!("Error in edit {}: old_string found {count} times", i + 1));
        }
        content = content.replacen(old, new, 1);
    }

    tokio::fs::write(&path, &content).await?;
    Ok(format!("Applied {} edits successfully", edits.len()))
}

async fn tool_grep(args: &Value, cwd: &str) -> Result<String> {
    let pattern = args["pattern"].as_str().unwrap_or("");
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(cwd);
    let path = resolve_path(path, cwd);
    let file_glob = args.get("glob").and_then(|v| v.as_str());
    let case_insensitive = args.get("case_insensitive").and_then(|v| v.as_bool()).unwrap_or(false);

    let mut cmd = Command::new("rg");
    cmd.arg("--no-heading").arg("-n").arg("--max-count").arg("50");
    if case_insensitive {
        cmd.arg("-i");
    }
    if let Some(g) = file_glob {
        cmd.arg("--glob").arg(g);
    }
    cmd.arg(pattern).arg(path.to_string_lossy().as_ref());

    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(15),
        cmd.output(),
    ).await {
        Ok(Ok(output)) => output,
        Ok(Err(_)) => {
            // rg not found, try grep
            let mut cmd2 = Command::new("grep");
            cmd2.arg("-rn");
            if case_insensitive {
                cmd2.arg("-i");
            }
            cmd2.arg(pattern).arg(path.to_string_lossy().as_ref());
            match tokio::time::timeout(std::time::Duration::from_secs(15), cmd2.output()).await {
                Ok(Ok(o)) => o,
                Ok(Err(e)) => return Ok(format!("Error: {e}")),
                Err(_) => return Ok("Error: grep timed out after 15s".to_string()),
            }
        }
        Err(_) => return Ok("Error: grep timed out after 15s".to_string()),
    };

    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    if text.len() > MAX_GREP_OUTPUT {
        text.truncate(MAX_GREP_OUTPUT);
        text.push_str("\n... (truncated)");
    }

    if text.trim().is_empty() {
        Ok("No matches found".to_string())
    } else {
        Ok(text)
    }
}

async fn tool_glob(args: &Value, cwd: &str) -> Result<String> {
    let pattern = args["pattern"].as_str().unwrap_or("");
    let base = args.get("path").and_then(|v| v.as_str()).unwrap_or(cwd);
    let base = resolve_path(base, cwd);
    let base_str = base.to_string_lossy().to_string();

    // Use a blocking task for glob
    let full_pattern = format!("{}/{}", base_str.trim_end_matches('/').trim_end_matches('\\'), pattern);
    let matches = tokio::task::spawn_blocking(move || {
        let mut results: Vec<String> = Vec::new();
        if let Ok(paths) = glob::glob(&full_pattern) {
            for entry in paths.flatten().take(MAX_GLOB_RESULTS) {
                if let Ok(rel) = entry.strip_prefix(&base) {
                    results.push(rel.to_string_lossy().to_string());
                } else {
                    results.push(entry.to_string_lossy().to_string());
                }
            }
        }
        results
    })
    .await?;

    if matches.is_empty() {
        Ok("No files matched".to_string())
    } else {
        Ok(matches.join("\n"))
    }
}

async fn tool_ls(args: &Value, cwd: &str) -> Result<String> {
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(cwd);
    let path = resolve_path(path, cwd);

    if !path.is_dir() {
        return Ok(format!("Error: Not a directory: {}", path.display()));
    }

    let mut entries: Vec<String> = Vec::new();
    let mut dir = tokio::fs::read_dir(&path).await?;
    while let Some(entry) = dir.next_entry().await? {
        let name = entry.file_name().to_string_lossy().to_string();
        let metadata = entry.metadata().await?;
        let prefix = if metadata.is_dir() { "d/" } else { "f/" };
        entries.push(format!("{prefix}{name}"));
    }
    entries.sort();

    if entries.is_empty() {
        Ok("(empty directory)".to_string())
    } else {
        Ok(entries.join("\n"))
    }
}

async fn tool_bash(args: &Value, cwd: &str) -> Result<String> {
    let command = args["command"].as_str().unwrap_or("");
    let timeout_secs = args.get("timeout").and_then(|v| v.as_u64()).unwrap_or(30);

    let shell = if cfg!(windows) { "cmd" } else { "sh" };
    let flag = if cfg!(windows) { "/C" } else { "-c" };

    let mut cmd = Command::new(shell);
    cmd.arg(flag)
        .arg(command)
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    match tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        cmd.output(),
    ).await {
        Ok(Ok(output)) => {
            let mut text = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                text.push_str(&stderr);
            }
            if text.len() > MAX_OUTPUT {
                text.truncate(MAX_OUTPUT);
                text.push_str("\n... (truncated)");
            }
            let exit_info = if !output.status.success() {
                format!("\n[exit code: {}]", output.status.code().unwrap_or(-1))
            } else {
                String::new()
            };
            let result = format!("{text}{exit_info}");
            if result.trim().is_empty() {
                Ok("(no output)".to_string())
            } else {
                Ok(result)
            }
        }
        Ok(Err(e)) => Ok(format!("Error: {e}")),
        Err(_) => Ok(format!("Error: Command timed out after {timeout_secs}s")),
    }
}

async fn tool_web_search(args: &Value) -> Result<String> {
    let query = args["query"].as_str().unwrap_or("");
    if query.is_empty() {
        return Ok("Error: query is required".to_string());
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let resp = client
        .get("https://html.duckduckgo.com/html/")
        .query(&[("q", query)])
        .header("User-Agent", "Mozilla/5.0")
        .send()
        .await?;

    let html = resp.text().await?;

    // Parse results from DDG HTML using simple regex-like parsing
    let mut results = Vec::new();
    let mut search_pos = 0;

    while results.len() < MAX_SEARCH_RESULTS {
        // Find result link
        let link_marker = "class=\"result__a\" href=\"";
        let link_start = match html[search_pos..].find(link_marker) {
            Some(pos) => search_pos + pos + link_marker.len(),
            None => break,
        };
        let link_end = match html[link_start..].find('"') {
            Some(pos) => link_start + pos,
            None => break,
        };
        let url = &html[link_start..link_end];

        // Find title (text after the href closing > until </a>)
        let title_start = match html[link_end..].find('>') {
            Some(pos) => link_end + pos + 1,
            None => break,
        };
        let title_end = match html[title_start..].find("</a>") {
            Some(pos) => title_start + pos,
            None => break,
        };
        let title = strip_html_tags(&html[title_start..title_end]);

        // Find snippet
        let snippet_marker = "class=\"result__snippet\"";
        let snippet_start = match html[title_end..].find(snippet_marker) {
            Some(pos) => {
                let s = title_end + pos;
                match html[s..].find('>') {
                    Some(p) => s + p + 1,
                    None => {
                        search_pos = title_end + 1;
                        continue;
                    }
                }
            }
            None => break,
        };
        let snippet_end = match html[snippet_start..].find("</a>") {
            Some(pos) => snippet_start + pos,
            None => break,
        };
        let snippet = strip_html_tags(&html[snippet_start..snippet_end]);

        results.push(format!("- {title}\n  {snippet}\n  {url}"));
        search_pos = snippet_end;
    }

    if results.is_empty() {
        Ok(format!("No results found for: {query}"))
    } else {
        Ok(format!("Search results for '{query}':\n\n{}", results.join("\n\n")))
    }
}

async fn tool_web_fetch(args: &Value) -> Result<String> {
    let url = args["url"].as_str().unwrap_or("");
    if url.is_empty() {
        return Ok("Error: url is required".to_string());
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let resp = client
        .get(url)
        .header("User-Agent", "Mozilla/5.0")
        .send()
        .await?;

    let html = resp.text().await?;

    // Strip scripts, styles, and tags
    let mut text = html.clone();
    // Remove script tags
    while let Some(start) = text.find("<script") {
        if let Some(end) = text[start..].find("</script>") {
            text = format!("{}{}", &text[..start], &text[start + end + 9..]);
        } else {
            break;
        }
    }
    // Remove style tags
    while let Some(start) = text.find("<style") {
        if let Some(end) = text[start..].find("</style>") {
            text = format!("{}{}", &text[..start], &text[start + end + 8..]);
        } else {
            break;
        }
    }
    // Remove HTML tags
    let text = strip_html_tags(&text);
    // Collapse whitespace
    let text: String = text
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");

    if text.is_empty() {
        return Ok(format!("No readable content found at {url}"));
    }

    let mut text = text;
    if text.len() > MAX_FETCH_CHARS {
        text.truncate(MAX_FETCH_CHARS);
        text.push_str("\n\n[...truncated]");
    }

    Ok(format!("Content from {url}:\n\n{text}"))
}

/// Strip HTML tags from a string.
fn strip_html_tags(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
            result.push(' ');
        } else if !in_tag {
            result.push(ch);
        }
    }
    result.trim().to_string()
}
