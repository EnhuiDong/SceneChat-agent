from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable


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
        temperature: float,
        max_tokens: int | None,
        native_json_mode: bool,
        token_limit_parameter: str = "max_tokens",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.native_json_mode = native_json_mode
        self.token_limit_parameter = token_limit_parameter
        self.extra_body = dict(extra_body or {})

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
            "temperature": self.temperature,
        }
        token_limit = self.max_tokens if max_tokens is None else max_tokens
        if token_limit is not None:
            request[self.token_limit_parameter] = token_limit
        if response_format and self.native_json_mode:
            request["response_format"] = response_format
        if self.extra_body:
            request["extra_body"] = dict(self.extra_body)

        response = self.client.chat.completions.create(**request)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("生成模型没有返回任何候选结果")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("生成模型响应缺少 message")
        return SimpleNamespace(
            content=response_content(getattr(message, "content", "")),
            raw=response,
        )
