"""
Registry of currently-connected WebSocket conversations.

Used by the scheduler to:
  - Find conversations for a given agent (so a scheduled task can be injected
    into a live session, or pushed as a notification)
  - Push out-of-band messages (e.g. "scheduled task fired")

Each conversation registers itself on connect and removes itself on disconnect.
"""

from typing import Any, Callable, Dict, List, Optional


class LiveConversations:
    def __init__(self) -> None:
        # stream_sid -> {agent_name, websocket, event_queue, send_text}
        self._sessions: Dict[str, dict] = {}

    def register(
        self,
        stream_sid: str,
        *,
        agent_name: str,
        websocket: Any,
        event_queue: Any,
        push_user_prompt: Callable[[str], Any],
    ) -> None:
        self._sessions[stream_sid] = {
            "stream_sid": stream_sid,
            "agent_name": agent_name,
            "websocket": websocket,
            "event_queue": event_queue,
            "push_user_prompt": push_user_prompt,
        }

    def update_agent(self, stream_sid: str, agent_name: str) -> None:
        s = self._sessions.get(stream_sid)
        if s:
            s["agent_name"] = agent_name

    def deregister(self, stream_sid: str) -> None:
        self._sessions.pop(stream_sid, None)

    def find_for_agent(self, agent_name: str) -> List[dict]:
        return [s for s in self._sessions.values() if s["agent_name"] == agent_name]

    def all(self) -> List[dict]:
        return list(self._sessions.values())


live_conversations = LiveConversations()
