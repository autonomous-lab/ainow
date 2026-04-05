"""
Pure state machine for ainow.

The process_event function is the heart of the system:
    (State, Event) -> (State, List[Action])

A trivial conversation controller (~30 lines of logic).
"""

from dataclasses import replace
from typing import List, Tuple

from .types import (
    AppState, Phase,
    Event, StreamStartEvent, StreamStopEvent, MediaEvent,
    StartOfTurnEvent, EndOfTurnEvent, AgentTurnDoneEvent,
    Action, FeedSTTAction, StartAgentTurnAction, ResetAgentTurnAction,
)


def process_event(state: AppState, event: Event) -> Tuple[AppState, List[Action]]:
    """
    Pure state machine: (State, Event) -> (State, Actions)

    Simple router:
    - MediaEvent        -> feed audio to STT
    - EndOfTurnEvent    -> start agent response
    - StartOfTurnEvent  -> interrupt (barge-in)
    - AgentTurnDoneEvent -> back to listening
    """
    if isinstance(event, StreamStartEvent):
        return replace(state, stream_sid=event.stream_sid, phase=Phase.LISTENING), []

    if isinstance(event, StreamStopEvent):
        actions: List[Action] = []
        if state.phase == Phase.RESPONDING:
            actions.append(ResetAgentTurnAction())
        return state, actions

    if isinstance(event, MediaEvent):
        return state, [FeedSTTAction(audio_bytes=event.audio_bytes)]

    if isinstance(event, EndOfTurnEvent):
        if event.transcript or event.images:
            if state.phase == Phase.RESPONDING:
                # Interrupt current generation and start new turn
                new_state = replace(state, phase=Phase.RESPONDING)
                return new_state, [ResetAgentTurnAction(), StartAgentTurnAction(transcript=event.transcript, images=event.images)]
            elif state.phase == Phase.LISTENING:
                new_state = replace(state, phase=Phase.RESPONDING)
                return new_state, [StartAgentTurnAction(transcript=event.transcript, images=event.images)]
        return state, []

    if isinstance(event, StartOfTurnEvent):
        if state.phase == Phase.RESPONDING:
            return replace(state, phase=Phase.LISTENING), [ResetAgentTurnAction()]
        return state, []

    if isinstance(event, AgentTurnDoneEvent):
        if state.phase == Phase.RESPONDING:
            return replace(state, phase=Phase.LISTENING), []
        return state, []

    return state, []
