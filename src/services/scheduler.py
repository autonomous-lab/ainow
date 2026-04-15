"""
Per-agent scheduled task scheduler.

Each agent's `agents/<name>/scheduled_tasks.json` lists tasks of the form:

  {
    "id": "task_<ts>_<rand>",
    "name": "Daily news digest",
    "schedule": "0 9 * * *"          # cron OR ISO ("2026-04-09 14:30")
    "prompt": "...",
    "enabled": true,
    "mode": "new_session" | "inject", # behavior when task fires
    "created_at": "...",
    "last_run": "...",
    "last_status": "ok" | "error: ...",
    "run_count": 7
  }

A single asyncio loop computes the soonest next-fire across all enabled tasks
across all agents and sleeps until then. When a task fires:

  - mode=new_session: spin up a fresh LLMService for that agent, run the
    prompt to completion, save the resulting conversation as a session under
    `agents/<name>/sessions/sched_<id>_<ts>.json`. Auto-approves dangerous
    tools (no human in the loop).

  - mode=inject: if there's a live WebSocket conversation for that agent,
    push the prompt onto its event queue (same code path as the user typing
    it). If no live conversation, fall back to new_session.

Either way, last_run/last_status/run_count are updated and a
`scheduled_task_fired` notification is pushed to any live conversation for
that agent.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from croniter import croniter

from ..log import get_logger
from . import agents
from .live_conversations import live_conversations

logger = get_logger("ainow.scheduler")


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


def _parse_iso(s: str) -> Optional[datetime]:
    """Try to parse an ISO date/datetime string. Returns None if not ISO."""
    if not _ISO_RE.match(s.strip()):
        return None
    try:
        # Allow both space and T separators, with or without seconds
        normalized = s.strip().replace("T", " ")
        if " " not in normalized:
            normalized += " 00:00:00"
        elif normalized.count(":") == 1:
            normalized += ":00"
        return datetime.fromisoformat(normalized.replace(" ", "T"))
    except Exception:
        return None


def is_valid_schedule(schedule: str) -> bool:
    """Validate a schedule string (cron or ISO date)."""
    if not schedule:
        return False
    iso = _parse_iso(schedule)
    if iso is not None:
        return True
    try:
        croniter(schedule, datetime.now())
        return True
    except Exception:
        return False


def next_fire_times(schedule: str, base: Optional[datetime] = None, n: int = 3) -> List[datetime]:
    """Return up to n next fire times for a schedule."""
    base = base or datetime.now()
    iso = _parse_iso(schedule)
    if iso is not None:
        return [iso] if iso > base else []
    try:
        it = croniter(schedule, base)
        return [it.get_next(datetime) for _ in range(n)]
    except Exception:
        return []


class SchedulerService:
    """Singleton background scheduler. One asyncio task drives everything."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self._running = False
        # In-memory cache of next-fire timestamps per (agent, task_id) so we
        # don't re-fire the same one-time task. Recomputed on reload.
        self._next_fire: Dict[Tuple[str, str], datetime] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._loop(), name="scheduler")
        logger.info("Scheduler service started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._wake.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
        logger.info("Scheduler service stopped")

    def reload(self) -> None:
        """Force the loop to re-read all task lists from disk and recompute."""
        self._wake.set()

    def reload_agent(self, agent_name: str) -> None:
        """Same as reload() — kept as a clearer hook from REST endpoints."""
        self._wake.set()

    async def run_now(self, agent_name: str, task_id: str) -> dict:
        """Execute a task immediately (used by the 'Run now' button)."""
        tasks = agents.read_scheduled_tasks(agent_name)
        task = next((t for t in tasks if t.get("id") == task_id), None)
        if not task:
            return {"ok": False, "error": "task not found"}
        result = await self._fire_task(agent_name, task, force=True)
        return result

    async def _loop(self) -> None:
        """Main scheduler loop. Sleeps until next earliest fire, then runs it."""
        while self._running:
            try:
                next_dt, fires = self._compute_next()
                if next_dt is None:
                    # No tasks scheduled. Sleep up to 60s until reload.
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=60.0)
                    except asyncio.TimeoutError:
                        pass
                    self._wake.clear()
                    continue

                now = datetime.now()
                wait = (next_dt - now).total_seconds()
                if wait > 0:
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=wait)
                        # Reloaded — recompute
                        self._wake.clear()
                        continue
                    except asyncio.TimeoutError:
                        pass

                # Fire all tasks scheduled for this moment
                for agent_name, task in fires:
                    asyncio.create_task(self._fire_task(agent_name, task))
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5)

    def _compute_next(self) -> Tuple[Optional[datetime], List[Tuple[str, dict]]]:
        """Walk all agents' tasks and find the soonest fire time + tasks at it."""
        now = datetime.now()
        soonest: Optional[datetime] = None
        fires: List[Tuple[str, dict]] = []

        for a in agents.list_agents():
            agent_name = a["name"]
            try:
                tasks = agents.read_scheduled_tasks(agent_name)
            except Exception:
                continue
            for t in tasks:
                if not t.get("enabled", True):
                    continue
                schedule = t.get("schedule", "")
                if not schedule:
                    continue
                fire_times = next_fire_times(schedule, base=now, n=1)
                if not fire_times:
                    continue
                ft = fire_times[0]
                if soonest is None or ft < soonest:
                    soonest = ft
                    fires = [(agent_name, t)]
                elif ft == soonest:
                    fires.append((agent_name, t))
        return soonest, fires

    async def _fire_task(self, agent_name: str, task: dict, force: bool = False) -> dict:
        """Execute one task. Updates last_run/last_status, fires notifications."""
        task_id = task.get("id", "?")
        task_name = task.get("name", "(unnamed)")
        mode = task.get("mode") or "new_session"
        prompt = task.get("prompt", "")
        logger.info(f"Firing task '{task_name}' ({task_id}) for agent '{agent_name}' mode={mode}")

        status = "ok"
        session_id: Optional[str] = None
        try:
            # Try inject mode first if requested AND there's a live conversation
            injected = False
            if mode == "inject":
                live = live_conversations.find_for_agent(agent_name)
                if live:
                    full = f"[SCHEDULED: {task_name}]\n{prompt}"
                    push = live[0].get("push_user_prompt")
                    if push:
                        await push(full)
                        injected = True
                        logger.info(f"Injected scheduled task '{task_name}' into live conversation")

            if not injected:
                # Headless: build a fresh LLMService for the agent and run the prompt
                session_id = await self._run_headless(agent_name, task)
        except Exception as e:
            status = f"error: {e}"
            logger.error(f"Task '{task_name}' failed: {e}")

        # Update task metadata
        try:
            agents.update_scheduled_task(agent_name, task_id, {
                "last_run": datetime.now().isoformat(),
                "last_status": status,
                "run_count": int(task.get("run_count", 0)) + 1,
            })
        except Exception as e:
            logger.error(f"Failed to update task metadata: {e}")

        # If it was a one-time task and it succeeded, disable it
        iso = _parse_iso(task.get("schedule", ""))
        if iso is not None and status == "ok":
            try:
                agents.update_scheduled_task(agent_name, task_id, {"enabled": False})
            except Exception:
                pass

        # Notify any live conversations for this agent
        await self._notify_fired(agent_name, task, status, session_id)

        # Wake the loop so it recomputes (since metadata changed)
        self._wake.set()
        return {"ok": status == "ok", "status": status, "session_id": session_id}

    async def _run_headless(self, agent_name: str, task: dict) -> Optional[str]:
        """Run the task prompt through a fresh LLMService and save as a session."""
        from .llm import LLMService
        from .tools import get_tool_schemas

        token_buf: List[str] = []
        done_event = asyncio.Event()

        async def on_token(t: str) -> None:
            token_buf.append(t)

        async def on_done() -> None:
            done_event.set()

        async def auto_approve(name: str, args: Any) -> bool:
            # No human in the loop — trust the agent's CLAUDE.md to be sane
            return True

        async def noop_tool_call(name: str, args: Any) -> None:
            pass

        async def noop_tool_result(call_id: str, name: str, result: str) -> None:
            pass

        llm = LLMService(
            on_token=on_token,
            on_done=on_done,
            tools=get_tool_schemas(),
            agent_name=agent_name,
            on_tool_call=noop_tool_call,
            on_tool_result=noop_tool_result,
            on_tool_confirm=auto_approve,
        )

        prompt = task.get("prompt", "")
        await llm.start(prompt)
        # Wait for completion (LLMService fires on_done from its internal loop)
        try:
            await asyncio.wait_for(done_event.wait(), timeout=600.0)
        except asyncio.TimeoutError:
            llm.cancel()
            raise RuntimeError("Headless task timed out after 10 minutes")

        # Save the conversation as a session with a recognizable id/title
        sched_id = task.get("id", "unknown")
        ts = int(datetime.now().timestamp())
        session_id = f"sched_{sched_id}_{ts}"
        try:
            llm.save_session(session_id)
        except Exception as e:
            logger.error(f"Failed to save scheduled session: {e}")
        return session_id

    async def _notify_fired(
        self,
        agent_name: str,
        task: dict,
        status: str,
        session_id: Optional[str],
    ) -> None:
        """Push a `scheduled_task_fired` WS message to any live conversations."""
        for s in live_conversations.find_for_agent(agent_name):
            ws = s.get("websocket")
            if not ws:
                continue
            try:
                await ws.send_text(json.dumps({
                    "type": "scheduled_task_fired",
                    "task_id": task.get("id"),
                    "task_name": task.get("name"),
                    "mode": task.get("mode") or "new_session",
                    "status": status,
                    "session_id": session_id,
                }))
            except Exception:
                pass


# Singleton
scheduler_service = SchedulerService()
