"""
External services for the AINow voice agent pipeline.

LLM        -- streaming chat completions
Whisper    -- local STT
Kokoro     -- local TTS
Fish       -- self-hosted TTS
"""

from .llm import LLMService

__all__ = [
    "LLMService",
]
