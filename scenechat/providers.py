import os
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from .errors import configuration_error, unsupported_provider_error


load_dotenv()

DASHSCOPE_MAX_EMBEDDING_BATCH_SIZE = 20


class SimulationLLMAdapter:
    """Expose the legacy ``complete`` interface over the JSON-capable client."""

    def __init__(self, client: Any):
        self.client = client
        self.model_name = getattr(client, "model_name", getattr(client, "model", "configured"))

    def complete(self, prompt: str, max_tokens: int = 360):
        response = self.client.invoke(
            [("user", prompt)],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return SimpleNamespace(text=str(content or ""))


def _required_env(name: str, description: str) -> str:
    value = os.getenv(name)
    if not value:
        raise configuration_error(name, description)
    return value


def get_generation_chat_model(
    temperature: float = 0.8,
    max_tokens: int | None = None,
):
    provider = os.getenv("MODEL_PROVIDER", "dashscope").lower()
    if provider != "dashscope":
        raise unsupported_provider_error(provider, "生成模型")

    api_key = _required_env("LLM_API_KEY", "生成模型 API Key")
    base_url = _required_env("LLM_API_BASE", "生成模型 API 地址")
    model_name = _required_env("LLM_MODEL", "生成模型名称")

    from langchain_openai import ChatOpenAI

    timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "180"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
    enable_thinking = os.getenv("LLM_ENABLE_THINKING", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        extra_body={"enable_thinking": enable_thinking},
    )


def get_simulation_llm():
    return SimulationLLMAdapter(get_generation_chat_model(temperature=0.65))


def get_embedding_model():
    provider = os.getenv(
        "EMBEDDING_PROVIDER", os.getenv("MODEL_PROVIDER", "dashscope")
    ).lower()
    model_name = os.getenv("EMBEDDING_MODEL")
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")

    if provider != "dashscope":
        raise unsupported_provider_error(provider, "向量模型")
    if not model_name:
        raise configuration_error("EMBEDDING_MODEL", "向量模型名称")
    if not api_key:
        raise configuration_error("EMBEDDING_API_KEY", "向量模型 API Key")

    from llama_index.embeddings.dashscope import DashScopeEmbedding

    # DashScope rejects text embedding requests containing more than 20 items.
    # The adapter's constructor default is 25 in the currently pinned version,
    # so this limit must be explicit rather than relying on library defaults.
    return DashScopeEmbedding(
        model_name=model_name,
        api_key=api_key,
        embed_batch_size=DASHSCOPE_MAX_EMBEDDING_BATCH_SIZE,
    )
