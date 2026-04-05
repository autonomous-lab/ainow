//! Agent -- self-contained LLM pipeline.
//!
//! Encapsulates the entire agent response lifecycle.
//! Owns conversation history across turns.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

use crate::llm::{LlmEvent, LlmService};
use crate::types::ImageAttachment;

/// Messages sent from the Agent to the conversation loop (WebSocket sender).
pub enum AgentOutput {
    /// Stream a text token to the browser.
    Token(String),
    /// Agent turn is complete.
    Done,
    /// Report a tool call to the browser.
    ToolCall { name: String, arguments: Value },
    /// Report a tool result to the browser.
    ToolResult { id: String, name: String, result: String },
    /// Pipeline stage update.
    Pipeline { stage: String, status: String, ms: u64 },
    /// Request browser tool execution.
    BrowserToolCall {
        request_id: String,
        name: String,
        arguments: Value,
        response_tx: tokio::sync::oneshot::Sender<String>,
    },
    /// Request tool confirmation from user.
    ToolConfirmRequest {
        confirm_id: String,
        name: String,
        arguments: Value,
        response_tx: tokio::sync::oneshot::Sender<bool>,
    },
}

pub struct Agent {
    /// Shared LLM service (persists history across turns).
    llm: Arc<Mutex<LlmService>>,
    /// Channel to send outputs to the conversation WebSocket loop.
    output_tx: mpsc::UnboundedSender<AgentOutput>,
    active: bool,
    t0: std::time::Instant,
    /// Handle to the event-forwarding task.
    forward_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Agent {
    pub fn new(
        base_url: String,
        api_key: String,
        model: String,
        system_prompt: String,
        output_tx: mpsc::UnboundedSender<AgentOutput>,
    ) -> Self {
        // Create a dummy event_tx; it gets replaced on each start_turn.
        let (event_tx, _) = mpsc::unbounded_channel();
        let llm = LlmService::new(base_url, api_key, model, system_prompt, event_tx);
        Self {
            llm: Arc::new(Mutex::new(llm)),
            output_tx,
            active: false,
            t0: std::time::Instant::now(),
            forward_handle: None,
        }
    }

    pub fn set_system_prompt(&self, prompt: String) {
        let llm = self.llm.clone();
        tokio::spawn(async move {
            llm.lock().await.set_system_prompt(&prompt);
        });
    }

    pub fn clear_history(&self) {
        let llm = self.llm.clone();
        tokio::spawn(async move {
            llm.lock().await.clear_history().await;
        });
    }

    pub fn restore_history(&self, messages: Vec<Value>) {
        let llm = self.llm.clone();
        tokio::spawn(async move {
            llm.lock().await.restore_history(&messages).await;
        });
    }

    /// Start a new agent turn.
    pub async fn start_turn(&mut self, transcript: &str, images: &[ImageAttachment]) {
        if self.active {
            self.cancel_turn().await;
        }

        self.active = true;
        self.t0 = std::time::Instant::now();

        // Create a new event channel for this turn
        let (event_tx, mut event_rx) = mpsc::unbounded_channel::<LlmEvent>();

        // Rebuild the LLM with the new event channel, preserving history
        {
            let mut llm = self.llm.lock().await;
            let history = llm.history.clone(); // Arc clone (same underlying data)
            let new_llm = LlmService::new(
                llm.base_url.clone(),
                llm.api_key.clone(),
                llm.model.clone(),
                llm.system_prompt.clone(),
                event_tx,
            );
            let old_cwd = llm.cwd.clone();
            *llm = new_llm;
            llm.history = history; // Reuse the same shared history
            llm.cwd = old_cwd;
            llm.start(transcript, images).await;
        }

        let _ = self.output_tx.send(AgentOutput::Pipeline {
            stage: "llm".to_string(),
            status: "active".to_string(),
            ms: 0,
        });

        // Forward LLM events to agent output
        let output_tx = self.output_tx.clone();
        let t0 = self.t0;
        self.forward_handle = Some(tokio::spawn(async move {
            let mut got_first = false;
            while let Some(event) = event_rx.recv().await {
                let ms = t0.elapsed().as_millis() as u64;
                match event {
                    LlmEvent::Token(token) => {
                        if !got_first {
                            got_first = true;
                            let _ = output_tx.send(AgentOutput::Pipeline {
                                stage: "llm".to_string(),
                                status: "active".to_string(),
                                ms,
                            });
                        }
                        let _ = output_tx.send(AgentOutput::Token(token));
                    }
                    LlmEvent::Done => {
                        let _ = output_tx.send(AgentOutput::Pipeline {
                            stage: "llm".to_string(),
                            status: "done".to_string(),
                            ms,
                        });
                        let _ = output_tx.send(AgentOutput::Done);
                        break;
                    }
                    LlmEvent::ToolCall { name, arguments } => {
                        let _ = output_tx.send(AgentOutput::ToolCall { name, arguments });
                    }
                    LlmEvent::ToolResult { id, name, result } => {
                        let _ = output_tx.send(AgentOutput::ToolResult { id, name, result });
                    }
                    LlmEvent::BrowserTool {
                        name,
                        arguments,
                        response_tx,
                    } => {
                        let request_id = uuid::Uuid::new_v4().to_string();
                        let _ = output_tx.send(AgentOutput::BrowserToolCall {
                            request_id,
                            name,
                            arguments,
                            response_tx,
                        });
                    }
                    LlmEvent::ToolConfirm {
                        name,
                        arguments,
                        response_tx,
                    } => {
                        let confirm_id = uuid::Uuid::new_v4().to_string();
                        let _ = output_tx.send(AgentOutput::ToolConfirmRequest {
                            confirm_id,
                            name,
                            arguments,
                            response_tx,
                        });
                    }
                }
            }
        }));
    }

    /// Cancel the current turn, preserve history.
    pub async fn cancel_turn(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;

        {
            self.llm.lock().await.cancel().await;
        }

        if let Some(handle) = self.forward_handle.take() {
            handle.abort();
            let _ = handle.await;
        }

        let _ = self.output_tx.send(AgentOutput::Pipeline {
            stage: "llm".to_string(),
            status: "idle".to_string(),
            ms: self.t0.elapsed().as_millis() as u64,
        });
    }

    /// Final cleanup when call ends.
    pub async fn cleanup(&mut self) {
        if self.active {
            self.cancel_turn().await;
        }
    }
}
