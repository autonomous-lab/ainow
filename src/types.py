"""
Type definitions for ainow.

All state, events, and actions are immutable dataclasses.
Minimal -- only what the main loop needs to route decisions.

Conversation history lives in Agent, not in AppState.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union, List


# =============================================================================
# STATE
# =============================================================================

class Phase(Enum):
    """Current phase of the conversation."""
    LISTENING = auto()    # Waiting for user / user speaking
    RESPONDING = auto()   # Agent active (LLM -> TTS -> Playback)


@dataclass(frozen=True)
class AppState:
    """
    Application state -- just routing information.

    Conversation history is owned by Agent, not tracked here.
    """
    phase: Phase = Phase.LISTENING
    stream_sid: Optional[str] = None


# =============================================================================
# EVENTS (inputs to the system)
# =============================================================================

@dataclass(frozen=True)
class StreamStartEvent:
    """Voice stream started."""
    stream_sid: str


@dataclass(frozen=True)
class StreamStopEvent:
    """Voice stream ended."""
    pass


@dataclass(frozen=True)
class MediaEvent:
    """Audio data received from client."""
    audio_bytes: bytes


@dataclass(frozen=True)
class StartOfTurnEvent:
    """User started speaking (barge-in)."""
    pass


@dataclass(frozen=True)
class EndOfTurnEvent:
    """User finished speaking."""
    transcript: str = ""
    images: tuple = ()  # Tuple of {data, mime} dicts for vision
    audio: Optional[str] = None  # base64 WAV for audio LLM mode


@dataclass(frozen=True)
class AgentTurnDoneEvent:
    """Agent finished speaking (playback complete)."""
    pass


Event = Union[
    StreamStartEvent, StreamStopEvent, MediaEvent,
    StartOfTurnEvent, EndOfTurnEvent,
    AgentTurnDoneEvent,
]


# =============================================================================
# ACTIONS (outputs from the system)
# =============================================================================

@dataclass(frozen=True)
class FeedSTTAction:
    """Send audio to STT service."""
    audio_bytes: bytes


@dataclass(frozen=True)
class StartAgentTurnAction:
    """Start agent response pipeline."""
    transcript: str = ""
    images: tuple = ()  # Tuple of {data, mime} dicts for vision
    audio: Optional[str] = None  # base64 WAV for audio LLM mode


@dataclass(frozen=True)
class ResetAgentTurnAction:
    """Cancel agent response and clear audio buffer."""
    pass


Action = Union[
    FeedSTTAction,
    StartAgentTurnAction,
    ResetAgentTurnAction,
]
