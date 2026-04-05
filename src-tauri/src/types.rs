//! Type definitions for AINow.
//!
//! All state, events, and actions are enums/structs.
//! Conversation history lives in the Agent/LLM, not in AppState.

use serde::{Deserialize, Serialize};

// =============================================================================
// STATE
// =============================================================================

/// Current phase of the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Listening,
    Responding,
}

/// Application state -- just routing information.
#[derive(Debug, Clone)]
pub struct AppState {
    pub phase: Phase,
    pub stream_sid: Option<String>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            phase: Phase::Listening,
            stream_sid: None,
        }
    }
}

// =============================================================================
// EVENTS (inputs to the system)
// =============================================================================

/// Image attachment for vision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAttachment {
    pub data: String,
    pub mime: String,
}

#[derive(Debug, Clone)]
pub enum Event {
    /// Voice stream started.
    StreamStart { stream_sid: String },
    /// Voice stream ended.
    StreamStop,
    /// User started speaking (barge-in).
    StartOfTurn,
    /// User finished speaking.
    EndOfTurn {
        transcript: String,
        images: Vec<ImageAttachment>,
    },
    /// Agent finished responding.
    AgentTurnDone,
}

// =============================================================================
// ACTIONS (outputs from the system)
// =============================================================================

#[derive(Debug, Clone)]
pub enum Action {
    /// Start agent response pipeline.
    StartAgentTurn {
        transcript: String,
        images: Vec<ImageAttachment>,
    },
    /// Cancel agent response.
    ResetAgentTurn,
}
