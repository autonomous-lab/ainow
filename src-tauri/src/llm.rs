//! LLM service with streaming, tool calling, and vision support.
//!
//! Uses reqwest to connect to an OpenAI-compatible API and streams
//! responses via SSE. Supports tool call loops and `<channel|>` filtering.

use anyhow::Result;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;

use crate::tools::{self, BROWSER_TOOLS, DANGEROUS_TOOLS};

/// Messages sent from the LLM service to the agent.
pub enum LlmEvent {
    /// A text token from the LLM.
    Token(String),
    /// LLM finished generating.
    Done,
    /// A tool call needs to be reported to the UI.
    ToolCall { name: String, arguments: Value },
    /// A tool result needs to be reported to the UI.
    ToolResult { id: String, name: String, result: String },
    /// A browser tool needs to be dispatched (returns result via channel).
    BrowserTool {
        name: String,
        arguments: Value,
        response_tx: tokio::sync::oneshot::Sender<String>,
    },
    /// A dangerous tool needs confirmation (returns bool via channel).
    ToolConfirm {
        name: String,
        arguments: Value,
        response_tx: tokio::sync::oneshot::Sender<bool>,
    },
}

/// Shared LLM state that persists across turns.
pub struct LlmService {
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) system_prompt: String,
    pub(crate) cwd: String,
    /// Conversation history, shared with the generation task.
    pub(crate) history: Arc<Mutex<Vec<Value>>>,
    tools: Vec<Value>,
    /// Channel to send events to the agent.
    event_tx: mpsc::UnboundedSender<LlmEvent>,
    /// Handle to the current generation task (if any).
    task_handle: Option<tokio::task::JoinHandle<()>>,
    /// Cancellation token.
    cancel: CancellationToken,
}

impl LlmService {
    pub fn new(
        base_url: String,
        api_key: String,
        model: String,
        system_prompt: String,
        event_tx: mpsc::UnboundedSender<LlmEvent>,
    ) -> Self {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());
        Self {
            base_url,
            api_key,
            model,
            system_prompt,
            cwd,
            history: Arc::new(Mutex::new(Vec::new())),
            tools: tools::tool_definitions(),
            event_tx,
            task_handle: None,
            cancel: CancellationToken::new(),
        }
    }

    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt = prompt.to_string();
    }

    pub async fn clear_history(&self) {
        self.history.lock().await.clear();
    }

    pub async fn restore_history(&self, messages: &[Value]) {
        let mut h = self.history.lock().await;
        h.clear();
        for msg in messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let text = msg.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if !text.is_empty() {
                match role {
                    "user" => h.push(json!({"role": "user", "content": text})),
                    "assistant" => h.push(json!({"role": "assistant", "content": text})),
                    _ => {}
                }
            }
        }
    }

    /// Start generating a response for the given user message.
    pub async fn start(&mut self, user_message: &str, images: &[crate::types::ImageAttachment]) {
        if self.task_handle.is_some() {
            self.cancel().await;
        }

        // Build user message content and append to history
        {
            let mut h = self.history.lock().await;
            if images.is_empty() {
                h.push(json!({"role": "user", "content": user_message}));
            } else {
                let mut content = vec![json!({"type": "text", "text": user_message})];
                for img in images {
                    content.push(json!({
                        "type": "image_url",
                        "image_url": {"url": &img.data}
                    }));
                }
                h.push(json!({"role": "user", "content": content}));
            }
        }

        let cancel = CancellationToken::new();
        self.cancel = cancel.clone();

        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let system_prompt = self.system_prompt.clone();
        let tools = self.tools.clone();
        let cwd = self.cwd.clone();
        let event_tx = self.event_tx.clone();
        let history = self.history.clone();

        let handle = tokio::spawn(async move {
            let result = generate_loop(
                &base_url,
                &api_key,
                &model,
                &system_prompt,
                history,
                &tools,
                &cwd,
                &event_tx,
                cancel,
            )
            .await;

            if let Err(e) = &result {
                log::error!("LLM generation failed: {e}");
            }

            let _ = event_tx.send(LlmEvent::Done);
        });

        self.task_handle = Some(handle);
    }

    pub async fn cancel(&mut self) {
        self.cancel.cancel();
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }
    }

    pub fn is_active(&self) -> bool {
        self.task_handle
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false)
    }
}

/// The main generation loop with tool call support.
async fn generate_loop(
    base_url: &str,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    history: Arc<Mutex<Vec<Value>>>,
    tools: &[Value],
    cwd: &str,
    event_tx: &mpsc::UnboundedSender<LlmEvent>,
    cancel: CancellationToken,
) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    // Build initial messages
    let mut messages = Vec::new();
    if !system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": system_prompt}));
    }
    {
        let h = history.lock().await;
        messages.extend(h.iter().cloned());
    }

    let mut final_content = String::new();

    loop {
        if cancel.is_cancelled() {
            break;
        }

        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "max_tokens": 2000,
            "temperature": 0.7
        });

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }

        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    if api_key.is_empty() {
                        "not-needed"
                    } else {
                        api_key
                    }
                ),
            )
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("LLM API error {status}: {body_text}"));
        }

        let mut content = String::new();
        let mut tool_calls_acc: HashMap<u64, ToolCallAcc> = HashMap::new();
        let mut finish_reason = String::new();
        let mut channel_buf = String::new();

        // Read SSE stream
        let mut stream = resp.bytes_stream();
        let mut line_buf = String::new();

        while let Some(chunk_result) = tokio::select! {
            chunk = stream.next() => chunk,
            _ = cancel.cancelled() => None,
        } {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("SSE stream error: {e}");
                    break;
                }
            };

            let chunk_str = String::from_utf8_lossy(&chunk);
            line_buf.push_str(&chunk_str);

            // Process complete SSE lines
            while let Some(newline_pos) = line_buf.find('\n') {
                let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }
                let data = &line[6..];
                if data == "[DONE]" {
                    continue;
                }

                let parsed: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let choices = parsed.get("choices").and_then(|v| v.as_array());
                let choice = choices.and_then(|c| c.first());
                let choice = match choice {
                    Some(c) => c,
                    None => continue,
                };

                let delta = choice.get("delta");
                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    finish_reason = fr.to_string();
                }

                // Stream text tokens with <channel|> filtering
                if let Some(delta_content) =
                    delta.and_then(|d| d.get("content")).and_then(|v| v.as_str())
                {
                    let mut text = delta_content.to_string();

                    // Prepend any leftover partial buffer
                    if !channel_buf.is_empty() {
                        text = format!("{}{}", channel_buf, text);
                        channel_buf.clear();
                    }

                    // Check for <channel|> marker
                    if text.contains("<channel|>") {
                        let after = text.split("<channel|>").last().unwrap_or("").to_string();
                        // Discard everything before marker (thinking)
                        content.clear();
                        if !after.is_empty() {
                            content.push_str(&after);
                            let _ = event_tx.send(LlmEvent::Token(after));
                        }
                    } else {
                        // Buffer trailing partial "<channel" across chunk boundaries
                        let tail_start = if text.len() >= 10 { text.len() - 10 } else { 0 };
                        let tail = &text[tail_start..];
                        if tail.contains('<') && content.is_empty() {
                            let cut = text[tail_start..]
                                .rfind('<')
                                .map(|p| tail_start + p)
                                .unwrap_or(text.len());
                            channel_buf = text[cut..].to_string();
                            let before = &text[..cut];
                            if !before.is_empty() {
                                content.push_str(before);
                                let _ = event_tx.send(LlmEvent::Token(before.to_string()));
                            }
                        } else if !text.is_empty() {
                            content.push_str(&text);
                            let _ = event_tx.send(LlmEvent::Token(text));
                        }
                    }
                }

                // Accumulate tool call fragments
                if let Some(tool_calls) = delta
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|v| v.as_array())
                {
                    for tc in tool_calls {
                        let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                        let entry =
                            tool_calls_acc
                                .entry(idx)
                                .or_insert_with(|| ToolCallAcc {
                                    id: String::new(),
                                    name: String::new(),
                                    arguments: String::new(),
                                });
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            entry.id = id.to_string();
                        }
                        if let Some(func) = tc.get("function") {
                            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                entry.name = name.to_string();
                            }
                            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                entry.arguments.push_str(args);
                            }
                        }
                    }
                }
            }
        }

        // Flush remaining channel buffer
        if !channel_buf.is_empty() {
            content.push_str(&channel_buf);
            let _ = event_tx.send(LlmEvent::Token(channel_buf));
        }

        if cancel.is_cancelled() {
            if !content.is_empty() {
                let mut h = history.lock().await;
                h.push(json!({"role": "assistant", "content": format!("{}...", content)}));
            }
            break;
        }

        // Check if we have tool calls to execute
        if finish_reason == "tool_calls"
            || (!tool_calls_acc.is_empty() && content.trim().is_empty())
        {
            let mut tool_calls_list = Vec::new();
            let mut indices: Vec<u64> = tool_calls_acc.keys().cloned().collect();
            indices.sort();
            for idx in indices {
                let tc = &tool_calls_acc[&idx];
                tool_calls_list.push(json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                    }
                }));
            }

            let assistant_msg = json!({
                "role": "assistant",
                "content": if content.is_empty() { Value::Null } else { Value::String(content.clone()) },
                "tool_calls": tool_calls_list
            });
            messages.push(assistant_msg);

            // Execute each tool call
            for tc_val in &tool_calls_list {
                let tc_name = tc_val["function"]["name"].as_str().unwrap_or("");
                let tc_args_str = tc_val["function"]["arguments"].as_str().unwrap_or("");
                let tc_id = tc_val["id"].as_str().unwrap_or("").to_string();

                let tc_args: Value = serde_json::from_str(tc_args_str).unwrap_or(json!({}));

                // Notify UI
                let _ = event_tx.send(LlmEvent::ToolCall {
                    name: tc_name.to_string(),
                    arguments: tc_args.clone(),
                });

                // Browser tools
                if BROWSER_TOOLS.contains(&tc_name) {
                    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                    let _ = event_tx.send(LlmEvent::BrowserTool {
                        name: tc_name.to_string(),
                        arguments: tc_args.clone(),
                        response_tx: resp_tx,
                    });

                    let result = match tokio::time::timeout(
                        std::time::Duration::from_secs(30),
                        resp_rx,
                    )
                    .await
                    {
                        Ok(Ok(r)) => r,
                        Ok(Err(_)) => "Error: Browser tool channel closed".to_string(),
                        Err(_) => "Error: Browser tool timed out after 30s".to_string(),
                    };

                    // capture_frame returns a data URL -- inject image
                    if tc_name == "capture_frame" && result.starts_with("data:") {
                        let image_msg = json!({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Here is the captured frame:"},
                                {"type": "image_url", "image_url": {"url": &result}}
                            ]
                        });
                        messages.push(image_msg.clone());
                        history.lock().await.push(image_msg);
                        let desc = "Frame captured. The image has been added to the conversation -- describe what you see.";
                        let _ = event_tx.send(LlmEvent::ToolResult {
                            id: tc_id.clone(),
                            name: tc_name.to_string(),
                            result: desc.to_string(),
                        });
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": desc
                        }));
                    } else {
                        let _ = event_tx.send(LlmEvent::ToolResult {
                            id: tc_id.clone(),
                            name: tc_name.to_string(),
                            result: result.clone(),
                        });
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result
                        }));
                    }
                    continue;
                }

                // Confirm dangerous tools
                if DANGEROUS_TOOLS.contains(&tc_name) {
                    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                    let _ = event_tx.send(LlmEvent::ToolConfirm {
                        name: tc_name.to_string(),
                        arguments: tc_args.clone(),
                        response_tx: resp_tx,
                    });

                    let approved = match tokio::time::timeout(
                        std::time::Duration::from_secs(60),
                        resp_rx,
                    )
                    .await
                    {
                        Ok(Ok(a)) => a,
                        _ => false,
                    };

                    if !approved {
                        let result = "Tool call denied by user.";
                        let _ = event_tx.send(LlmEvent::ToolResult {
                            id: tc_id.clone(),
                            name: tc_name.to_string(),
                            result: result.to_string(),
                        });
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result
                        }));
                        continue;
                    }
                }

                // Execute tool
                let result = tools::execute_tool(tc_name, &tc_args, cwd).await;

                // Notify UI (truncated for display)
                let truncated = if result.len() > 2000 {
                    result[..2000].to_string()
                } else {
                    result.clone()
                };
                let _ = event_tx.send(LlmEvent::ToolResult {
                    id: tc_id.clone(),
                    name: tc_name.to_string(),
                    result: truncated,
                });

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result
                }));
            }

            // Continue loop -- LLM will respond after seeing tool results
            continue;
        } else {
            // Normal text completion -- done
            final_content = content;
            break;
        }
    }

    // Update history with final assistant message
    if !final_content.is_empty() {
        let mut h = history.lock().await;
        h.push(json!({"role": "assistant", "content": final_content}));
    }

    Ok(())
}

struct ToolCallAcc {
    id: String,
    name: String,
    arguments: String,
}
