from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable
import asyncio

from .build_control import CURRENT_BUILD, await_controlled
from .config import config_int


def create_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    timeout: float,
    max_retries: int,
):
    """Create one OpenAI transport for OpenAI and compatible endpoints."""
    from openai import OpenAI

    options: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if base_url:
        options["base_url"] = base_url
    return OpenAI(**options)


def response_content(response: Any) -> str:
    """Normalize OpenAI-compatible text content into one string."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if isinstance(message, (tuple, list)) and len(message) == 2:
        return {"role": str(message[0]), "content": str(message[1])}
    role = getattr(message, "role", None) or getattr(message, "type", None)
    content = getattr(message, "content", None)
    if role is None or content is None:
        raise TypeError("消息必须是 role/content 映射、二元组或消息对象")
    return {"role": str(role), "content": content}


class OpenAICompatibleChatModel:
    """Small chat interface shared by OpenAI-compatible model providers."""

    def __init__(
        self,
        *,
        client: Any,
        model_name: str,
        temperature: float | None,
        max_tokens: int | None,
        native_json_mode: bool,
        token_limit_parameter: str = "max_tokens",
        extra_body: dict[str, Any] | None = None,
        reasoning_token_reserve: int = 0,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.native_json_mode = native_json_mode
        self.token_limit_parameter = token_limit_parameter
        self.extra_body = dict(extra_body or {})
        self.reasoning_token_reserve = reasoning_token_reserve

    def invoke(
        self,
        messages: Iterable[Any],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ):
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [_message_dict(message) for message in messages],
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        token_limit = self.max_tokens if max_tokens is None else max_tokens
        if token_limit is not None:
            request[self.token_limit_parameter] = token_limit + self.reasoning_token_reserve
        if response_format and self.native_json_mode:
            request["response_format"] = response_format
        if self.extra_body:
            request["extra_body"] = dict(self.extra_body)

        control = CURRENT_BUILD.get()
        if control is None:
            response = self.client.chat.completions.create(**request)
        else:
            response = asyncio.run(self._controlled_request(request, control))
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("生成模型没有返回任何候选结果")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("生成模型响应缺少 message")
        return SimpleNamespace(
            content=response_content(getattr(message, "content", "")),
            raw=response,
            finish_reason=getattr(choices[0], "finish_reason", ""),
        )

    async def _controlled_request(self, request, control):
        from openai import AsyncOpenAI

        control.check()
        timeout = config_int("scenario", "request_timeout_seconds", 120, minimum=1, maximum=600)
        configured_read_timeout = getattr(self.client.timeout, "read", None)
        if configured_read_timeout is not None:
            timeout = min(timeout, configured_read_timeout)
        # No SDK retries here: structured generation owns the sole retry loop.
        async with AsyncOpenAI(
            api_key=self.client.api_key, base_url=str(self.client.base_url),
            timeout=timeout, max_retries=0,
        ) as client:
            return await await_controlled(client.chat.completions.create(**request), control, timeout)

    def close(self):
        self.client.close()
