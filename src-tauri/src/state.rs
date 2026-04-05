//! Pure state machine for AINow.
//!
//! The `process_event` function is the heart of the system:
//!     (State, Event) -> (State, Vec<Action>)

use crate::types::{Action, AppState, Event, Phase};

/// Pure state machine: (State, Event) -> (State, Actions)
pub fn process_event(state: &AppState, event: &Event) -> (AppState, Vec<Action>) {
    match event {
        Event::StreamStart { stream_sid } => {
            let new_state = AppState {
                stream_sid: Some(stream_sid.clone()),
                phase: Phase::Listening,
            };
            (new_state, vec![])
        }

        Event::StreamStop => {
            let mut actions = vec![];
            if state.phase == Phase::Responding {
                actions.push(Action::ResetAgentTurn);
            }
            (state.clone(), actions)
        }

        Event::EndOfTurn {
            transcript,
            images,
        } => {
            if !transcript.is_empty() || !images.is_empty() {
                let mut actions = vec![];
                if state.phase == Phase::Responding {
                    actions.push(Action::ResetAgentTurn);
                }
                actions.push(Action::StartAgentTurn {
                    transcript: transcript.clone(),
                    images: images.clone(),
                });
                let new_state = AppState {
                    phase: Phase::Responding,
                    ..state.clone()
                };
                (new_state, actions)
            } else {
                (state.clone(), vec![])
            }
        }

        Event::StartOfTurn => {
            if state.phase == Phase::Responding {
                let new_state = AppState {
                    phase: Phase::Listening,
                    ..state.clone()
                };
                (new_state, vec![Action::ResetAgentTurn])
            } else {
                (state.clone(), vec![])
            }
        }

        Event::AgentTurnDone => {
            if state.phase == Phase::Responding {
                let new_state = AppState {
                    phase: Phase::Listening,
                    ..state.clone()
                };
                (new_state, vec![])
            } else {
                (state.clone(), vec![])
            }
        }
    }
}
