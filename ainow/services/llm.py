"""
LLM service with streaming, tool calling, and vision support.
"""

import os
import json
import asyncio
from typing import Optional, Callable, Awaitable, List, Dict, Any

from openai import AsyncOpenAI, NOT_GIVEN

from ..log import ServiceLogger

log = ServiceLogger("LLM")

SYSTEM_PROMPT = ""


class LLMService:
    """
    OpenAI streaming LLM service with tool calling and vision.

    Manages conversation history and streams tokens via callback.
    Supports a tool call loop: LLM -> tool_calls -> execute -> append results -> LLM again.
    """

    def __init__(
        self,
        on_token: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        cwd: Optional[str] = None,
        on_tool_call: Optional[Callable[[str, Any], Awaitable[None]]] = None,
        on_tool_result: Optional[Callable[[str, str, str], Awaitable[None]]] = None,
        system_prompt: Optional[str] = None,
        on_tool_confirm: Optional[Callable[[str, Any], Awaitable[bool]]] = None,
        on_browser_tool: Optional[Callable[[str, dict], Awaitable[str]]] = None,
    ):
        self._on_token = on_token
        self._on_done = on_done
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result
        self._system_prompt = system_prompt or SYSTEM_PROMPT
        self._on_tool_confirm = on_tool_confirm
        self._on_browser_tool = on_browser_tool

        self._base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self._api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
        self._model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self._tools = tools
        self._cwd = cwd or os.getcwd()

        self._client = AsyncOpenAI(
            api_key=self._api_key if self._api_key else "not-needed",
            base_url=self._base_url,
        )
        self._task: Optional[asyncio.Task] = None
        self._running = False

        self._history: List[Dict] = []

    @property
    def is_active(self) -> bool:
        return self._running and self._task is not None

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value if value else SYSTEM_PROMPT

    @property
    def history(self) -> List[Dict]:
        return self._history.copy()

    def clear_history(self) -> None:
        self._history = []

    def restore_history(self, messages: list) -> None:
        """Restore history from client-side chat messages."""
        self._history = []
        for msg in messages:
            role = msg.get("role", "")
            text = msg.get("text", "")
            if role == "user" and text:
                self._history.append({"role": "user", "content": text})
            elif role == "assistant" and text:
                self._history.append({"role": "assistant", "content": text})

    async def start(self, user_message: str, images=None) -> None:
        """Start generating a response. Supports optional images for vision."""
        if self._running:
            await self.cancel()

        # Build user message content
        if images:
            content = [{"type": "text", "text": user_message}]
            for img in images:
                data_url = img["data"]
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
                log.info(f"Media attached: mime={img.get('mime','?')}, data_url_len={len(data_url)}")
            self._history.append({"role": "user", "content": content})
            log.info(f"Multimodal message: text='{user_message[:50]}', {len(images)} file(s)")
        else:
            self._history.append({"role": "user", "content": user_message})

        self._running = True
        self._task = asyncio.create_task(self._generate())
        log.connected()

    async def cancel(self) -> None:
        """Cancel ongoing generation."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        log.cancelled()

    def inject_image(self, data_url: str) -> None:
        """Inject an image into conversation history so the LLM can see it."""
        self._history.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })

    def _strip_old_media(self) -> None:
        """Replace heavy media (images/videos) in older history with text placeholders.
        Keeps media only in the most recent user message so old uploads aren't resent."""
        for i, msg in enumerate(self._history[:-2]):  # skip last 2 (latest user + assistant)
            if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            stripped = []
            had_media = False
            for block in msg["content"]:
                if block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    mime = "video" if url.startswith("data:video") else "image"
                    stripped.append({"type": "text", "text": f"[{mime} was shared earlier]"})
                    had_media = True
                else:
                    stripped.append(block)
            if had_media:
                self._history[i] = {"role": "user", "content": stripped}

    async def _generate(self) -> None:
        """Generate response with tool call loop."""
        from .tools import execute_tool, DANGEROUS_TOOLS, BROWSER_TOOLS

        final_content = ""

        try:
            messages = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages += self._history

            had_tool_calls = False
            while True:  # Tool call loop
                if not self._running:
                    break

                create_kwargs = dict(
                    model=self._model,
                    messages=messages,
                    stream=True,
                    max_tokens=2000,
                    temperature=0.7,
                )
                if self._tools:
                    create_kwargs["tools"] = self._tools

                stream = await self._client.chat.completions.create(**create_kwargs)

                content = ""
                tool_calls_acc: Dict[int, Dict] = {}  # index -> {id, name, arguments}
                finish_reason = None
                _channel_buf = ""  # Small buffer to catch <channel|> at stream start

                async for chunk in stream:
                    if not self._running:
                        break

                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue

                    delta = choice.delta
                    finish_reason = choice.finish_reason

                    # Stream text tokens — filter <channel|> marker
                    if delta and delta.content:
                        text = delta.content
                        # Prepend any leftover partial buffer
                        if _channel_buf:
                            text = _channel_buf + text
                            _channel_buf = ""
                        # Check for <channel|> marker
                        if "<channel|>" in text:
                            _, _, after = text.partition("<channel|>")
                            # Discard everything before marker (thinking)
                            content = ""
                            if after:
                                content += after
                                await self._on_token(after)
                        else:
                            # Buffer trailing partial "<channel" across chunk boundaries
                            tail = text[-10:] if len(text) >= 10 else text
                            if "<" in tail and not content:
                                cut = text.rfind("<", max(0, len(text) - 10))
                                _channel_buf = text[cut:]
                                text = text[:cut]
                            if text:
                                content += text
                                await self._on_token(text)

                    # Accumulate tool call fragments
                    if delta and delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc.id or "",
                                    "name": (tc.function.name if tc.function else "") or "",
                                    "arguments": "",
                                }
                            if tc.id:
                                tool_calls_acc[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_acc[idx]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls_acc[idx]["arguments"] += tc.function.arguments

                # Flush any remaining partial buffer
                if _channel_buf:
                    content += _channel_buf
                    await self._on_token(_channel_buf)
                    _channel_buf = ""

                if not self._running:
                    if content:
                        self._history.append({"role": "assistant", "content": content + "..."})
                    break

                # Check if we have tool calls to execute
                if finish_reason == "tool_calls" or (tool_calls_acc and not content.strip()):
                    # Build the assistant message with tool_calls
                    tool_calls_list = []
                    for idx in sorted(tool_calls_acc.keys()):
                        tc_data = tool_calls_acc[idx]
                        tool_calls_list.append({
                            "id": tc_data["id"],
                            "type": "function",
                            "function": {
                                "name": tc_data["name"],
                                "arguments": tc_data["arguments"],
                            },
                        })

                    assistant_msg = {"role": "assistant", "content": content or None, "tool_calls": tool_calls_list}
                    messages.append(assistant_msg)

                    # Execute each tool call
                    had_tool_calls = True
                    for tc in tool_calls_list:
                        tc_name = tc["function"]["name"]
                        tc_args_str = tc["function"]["arguments"]
                        tc_id = tc["id"]

                        try:
                            tc_args = json.loads(tc_args_str) if tc_args_str else {}
                        except json.JSONDecodeError:
                            tc_args = {}

                        # Notify UI
                        if self._on_tool_call:
                            await self._on_tool_call(tc_name, tc_args)

                        # Browser tools — dispatch to browser via callback
                        if tc_name in BROWSER_TOOLS and self._on_browser_tool:
                            try:
                                result = await self._on_browser_tool(tc_name, tc_args)
                            except Exception as e:
                                result = f"Error: {e}"

                            # capture_frame returns a data URL — inject image so LLM can see it
                            if tc_name == "capture_frame" and result.startswith("data:"):
                                image_msg = {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Here is the captured frame:"},
                                        {"type": "image_url", "image_url": {"url": result}},
                                    ],
                                }
                                messages.append(image_msg)
                                self._history.append(image_msg)
                                result = "Frame captured. The image has been added to the conversation — describe what you see."

                            if self._on_tool_result:
                                await self._on_tool_result(tc_id, tc_name, result)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": result,
                            })
                            continue

                        # Confirm dangerous tools
                        if tc_name in DANGEROUS_TOOLS and self._on_tool_confirm:
                            approved = await self._on_tool_confirm(tc_name, tc_args)
                            if not approved:
                                result = "Tool call denied by user."
                                if self._on_tool_result:
                                    await self._on_tool_result(tc_id, tc_name, result)
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc_id,
                                    "content": result,
                                })
                                continue

                        # Execute
                        result = await execute_tool(tc_name, tc_args, cwd=self._cwd)

                        # Notify UI
                        if self._on_tool_result:
                            await self._on_tool_result(tc_id, tc_name, result)

                        # Append tool result to messages for next iteration
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result,
                        })

                    # Continue loop — LLM will respond after seeing tool results
                    continue
                else:
                    # Normal text completion — done
                    final_content = content
                    break

            if self._running and final_content:
                self._history.append({"role": "assistant", "content": final_content})
                self._strip_old_media()
                await self._on_done()
            elif self._running:
                self._strip_old_media()
                await self._on_done()

        except asyncio.CancelledError:
            if final_content:
                self._history.append({"role": "assistant", "content": final_content + "..."})
            raise

        except Exception as e:
            log.error("Generation failed", e)
            await self._on_done()

        finally:
            self._running = False
            self._task = None
