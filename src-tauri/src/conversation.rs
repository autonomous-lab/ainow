//! The main event loop for AINow.
//!
//! Drives the state machine:
//!     while connected:
//!         event = receive()
//!         state, actions = process_event(state, event)
//!         for action in actions: dispatch(action)

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde_json::json;
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};

use crate::agent::{Agent, AgentOutput};
use crate::settings::Settings;
use crate::state::process_event;
use crate::types::*;

/// Control messages from the WebSocket reader that bypass the state machine.
enum ControlMsg {
    SetSystemPrompt(String),
    ClearSession,
    RestoreHistory(Vec<serde_json::Value>),
    SetLang(String),
    ToolConfirmResponse { confirm_id: String, approved: bool },
    BrowserToolResult { request_id: String, result: String },
}

/// Run the conversation loop over a WebSocket connection.
pub async fn run_conversation(ws: WebSocket, settings: Settings) {
    let (mut ws_tx, mut ws_rx) = ws.split();
    let stream_sid = uuid::Uuid::new_v4().to_string();

    // Event queue for the state machine
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<Event>();

    // Control message channel (for messages that bypass the state machine)
    let (ctrl_tx, mut ctrl_rx) = mpsc::unbounded_channel::<ControlMsg>();

    // Agent output channel
    let (agent_output_tx, mut agent_output_rx) = mpsc::unbounded_channel::<AgentOutput>();

    // Pending browser tool / confirm futures
    let mut pending_browser_tools: HashMap<String, oneshot::Sender<String>> = HashMap::new();
    let mut pending_confirms: HashMap<String, oneshot::Sender<bool>> = HashMap::new();

    let custom_system_prompt = settings.system_prompt.clone();

    // Create agent
    let mut agent = Agent::new(
        settings.llm_base_url.clone(),
        settings.llm_api_key.clone(),
        settings.llm_model.clone(),
        custom_system_prompt.clone(),
        agent_output_tx,
    );

    // Send config to browser
    let config_msg = json!({
        "type": "config",
        "sample_rate": 16000,
        "use_browser_stt": true,
        "use_browser_tts": true,
        "system_prompt": &custom_system_prompt,
    });
    if ws_tx
        .send(Message::Text(config_msg.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    // Initialize state machine
    let mut state = AppState {
        stream_sid: Some(stream_sid.clone()),
        ..AppState::default()
    };

    // Background task: read from WebSocket and route to event_tx / ctrl_tx
    let event_tx_reader = event_tx.clone();
    let ctrl_tx_reader = ctrl_tx;
    let reader_handle = tokio::spawn(async move {
        while let Some(msg) = ws_rx.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    let data: serde_json::Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let ctrl_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match ctrl_type {
                        "end_of_turn" => {
                            let transcript = data
                                .get("transcript")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .trim()
                                .to_string();
                            let images: Vec<ImageAttachment> = data
                                .get("images")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| serde_json::from_value(v.clone()).ok())
                                        .collect()
                                })
                                .unwrap_or_default();
                            if !transcript.is_empty() || !images.is_empty() {
                                let _ =
                                    event_tx_reader.send(Event::EndOfTurn { transcript, images });
                            }
                        }
                        "start_of_turn" => {
                            let _ = event_tx_reader.send(Event::StartOfTurn);
                        }
                        "set_system_prompt" => {
                            let prompt = data
                                .get("prompt")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let _ = ctrl_tx_reader.send(ControlMsg::SetSystemPrompt(prompt));
                        }
                        "clear_session" => {
                            let _ = ctrl_tx_reader.send(ControlMsg::ClearSession);
                        }
                        "restore_history" => {
                            let messages = data
                                .get("messages")
                                .and_then(|v| v.as_array())
                                .cloned()
                                .unwrap_or_default();
                            let _ = ctrl_tx_reader.send(ControlMsg::RestoreHistory(messages));
                        }
                        "set_lang" => {
                            let lang = data
                                .get("lang")
                                .and_then(|v| v.as_str())
                                .unwrap_or("en-US")
                                .to_string();
                            let _ = ctrl_tx_reader.send(ControlMsg::SetLang(lang));
                        }
                        "tool_confirm_response" => {
                            let confirm_id = data
                                .get("confirm_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let approved = data
                                .get("approved")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            let _ = ctrl_tx_reader
                                .send(ControlMsg::ToolConfirmResponse { confirm_id, approved });
                        }
                        "browser_tool_result" => {
                            let request_id = data
                                .get("request_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let result = data
                                .get("result")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let _ = ctrl_tx_reader
                                .send(ControlMsg::BrowserToolResult { request_id, result });
                        }
                        "stop" => {
                            let _ = event_tx_reader.send(Event::StreamStop);
                            break;
                        }
                        _ => {}
                    }
                }
                Ok(Message::Close(_)) | Err(_) => {
                    let _ = event_tx_reader.send(Event::StreamStop);
                    break;
                }
                _ => {}
            }
        }
    });

    // Main event loop
    loop {
        tokio::select! {
            // State machine events
            Some(event) = event_rx.recv() => {
                let old_phase = state.phase;
                let (new_state, actions) = process_event(&state, &event);
                state = new_state;

                // Send phase changes to browser
                if old_phase != state.phase {
                    let phase_str = match state.phase {
                        Phase::Listening => "listening",
                        Phase::Responding => "responding",
                    };
                    let _ = ws_tx
                        .send(Message::Text(json!({"type": "state", "phase": phase_str}).to_string().into()))
                        .await;
                }

                // Dispatch actions
                for action in actions {
                    match action {
                        Action::StartAgentTurn { transcript, images } => {
                            agent.start_turn(&transcript, &images).await;
                        }
                        Action::ResetAgentTurn => {
                            agent.cancel_turn().await;
                            let _ = ws_tx
                                .send(Message::Text(json!({"type": "clear"}).to_string().into()))
                                .await;
                        }
                    }
                }

                // Exit check
                if matches!(event, Event::StreamStop) {
                    break;
                }
            }

            // Control messages (bypass state machine)
            Some(ctrl) = ctrl_rx.recv() => {
                match ctrl {
                    ControlMsg::SetSystemPrompt(prompt) => {
                        agent.set_system_prompt(prompt);
                        log::info!("System prompt updated");
                    }
                    ControlMsg::ClearSession => {
                        agent.clear_history();
                        log::info!("Session cleared");
                    }
                    ControlMsg::RestoreHistory(messages) => {
                        agent.restore_history(messages);
                        log::info!("History restored");
                    }
                    ControlMsg::SetLang(lang) => {
                        log::info!("Language set to: {lang}");
                    }
                    ControlMsg::ToolConfirmResponse { confirm_id, approved } => {
                        if let Some(tx) = pending_confirms.remove(&confirm_id) {
                            let _ = tx.send(approved);
                        }
                    }
                    ControlMsg::BrowserToolResult { request_id, result } => {
                        if let Some(tx) = pending_browser_tools.remove(&request_id) {
                            let _ = tx.send(result);
                        }
                    }
                }
            }

            // Agent output (tokens, tool calls, etc.)
            Some(output) = agent_output_rx.recv() => {
                match output {
                    AgentOutput::Token(token) => {
                        let _ = ws_tx
                            .send(Message::Text(json!({"type": "transcript", "token": token}).to_string().into()))
                            .await;
                    }
                    AgentOutput::Done => {
                        let _ = ws_tx
                            .send(Message::Text(json!({"type": "tts_done"}).to_string().into()))
                            .await;
                        let _ = event_tx.send(Event::AgentTurnDone);
                    }
                    AgentOutput::ToolCall { name, arguments } => {
                        let _ = ws_tx
                            .send(Message::Text(json!({
                                "type": "tool_call",
                                "name": name,
                                "arguments": arguments
                            }).to_string().into()))
                            .await;
                    }
                    AgentOutput::ToolResult { id, name, result } => {
                        let _ = ws_tx
                            .send(Message::Text(json!({
                                "type": "tool_result",
                                "id": id,
                                "name": name,
                                "result": result
                            }).to_string().into()))
                            .await;
                    }
                    AgentOutput::Pipeline { stage, status, ms } => {
                        let _ = ws_tx
                            .send(Message::Text(json!({
                                "type": "pipeline",
                                "stage": stage,
                                "status": status,
                                "ms": ms
                            }).to_string().into()))
                            .await;
                    }
                    AgentOutput::BrowserToolCall { request_id, name, arguments, response_tx } => {
                        pending_browser_tools.insert(request_id.clone(), response_tx);
                        let _ = ws_tx
                            .send(Message::Text(json!({
                                "type": "browser_tool_call",
                                "request_id": request_id,
                                "name": name,
                                "arguments": arguments
                            }).to_string().into()))
                            .await;
                    }
                    AgentOutput::ToolConfirmRequest { confirm_id, name, arguments, response_tx } => {
                        pending_confirms.insert(confirm_id.clone(), response_tx);
                        let _ = ws_tx
                            .send(Message::Text(json!({
                                "type": "tool_confirm",
                                "confirm_id": confirm_id,
                                "name": name,
                                "arguments": arguments
                            }).to_string().into()))
                            .await;
                    }
                }
            }

            else => break,
        }
    }

    // Cleanup
    for (_, tx) in pending_confirms.drain() {
        let _ = tx.send(false);
    }
    for (_, tx) in pending_browser_tools.drain() {
        let _ = tx.send("Error: Connection closed".to_string());
    }

    agent.cleanup().await;
    reader_handle.abort();
    let _ = reader_handle.await;

    log::info!("Conversation ended (stream_sid={stream_sid})");
}
