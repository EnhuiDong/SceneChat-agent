from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from .errors import SceneChatError, configuration_error, unsupported_provider_error
from .openai_compat import OpenAICompatibleChatModel, create_openai_client


load_dotenv()

DASHSCOPE_COMPATIBLE_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_MAX_EMBEDDING_BATCH_SIZE = 20
OPENAI_DEFAULT_EMBEDDING_BATCH_SIZE = 100
COMPATIBLE_DEFAULT_EMBEDDING_BATCH_SIZE = 20

CHAT_PROVIDERS = {"dashscope", "openai", "openai_compatible"}
EMBEDDING_PROVIDERS = {
    "dashscope",
    "dashscope_native",
    "dashscope_compatible",
    "openai",
    "openai_compatible",
}


class SimulationLLMAdapter:
    """Expose the legacy ``complete`` interface over the shared chat client."""

    def __init__(self, client: Any):
        self.client = client
        self.model_name = getattr(
            client, "model_name", getattr(client, "model", "configured")
        )

    def complete(self, prompt: str, max_tokens: int = 360):
        response = self.client.invoke(
            [("user", prompt)],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return SimpleNamespace(text=str(getattr(response, "content", "") or ""))


def _normalized_provider(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _required_env(name: str, description: str) -> str:
    value = os.getenv(name)
    if not value:
        raise configuration_error(name, description)
    return value


def _invalid_env(name: str, description: str) -> SceneChatError:
    return SceneChatError(
        "model_configuration_invalid",
        f"后端配置的{description}（{name}）无效，请完善 .env 后重新启动服务。",
        stage="preflight",
        status_code=503,
    )


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise _invalid_env(name, "布尔值")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise _invalid_env(name, "整数") from exc
    if value < minimum:
        raise _invalid_env(name, f"不小于 {minimum} 的整数")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise _invalid_env(name, "数字") from exc
    if value < minimum:
        raise _invalid_env(name, f"不小于 {minimum} 的数字")
    return value


def _chat_provider() -> str:
    # MODEL_PROVIDER remains a backwards-compatible alias for existing .env files.
    provider = _normalized_provider(
        os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or "dashscope"
    )
    if provider not in CHAT_PROVIDERS:
        raise unsupported_provider_error(provider, "生成模型")
    return provider


def _json_mode(provider: str) -> bool:
    raw = os.getenv("LLM_JSON_MODE", "auto").strip().lower()
    if raw == "auto":
        return provider in {"dashscope", "openai"}
    if raw in {"native", "on", "true", "1"}:
        return True
    if raw in {"prompt", "off", "false", "0"}:
        return False
    raise _invalid_env("LLM_JSON_MODE", "JSON 模式")


def _token_limit_parameter() -> str:
    value = os.getenv("LLM_TOKEN_LIMIT_PARAMETER", "max_tokens").strip()
    if value not in {"max_tokens", "max_completion_tokens"}:
        raise _invalid_env("LLM_TOKEN_LIMIT_PARAMETER", "输出 token 参数名")
    return value


def get_generation_chat_model(
    temperature: float = 0.8,
    max_tokens: int | None = None,
):
    provider = _chat_provider()
    api_key = _required_env("LLM_API_KEY", "生成模型 API Key")
    model_name = _required_env("LLM_MODEL", "生成模型名称")

    configured_base_url = (os.getenv("LLM_API_BASE") or "").strip() or None
    if provider == "dashscope":
        base_url = configured_base_url or DASHSCOPE_COMPATIBLE_BASE_URL
    elif provider == "openai_compatible":
        base_url = configured_base_url or _required_env(
            "LLM_API_BASE", "OpenAI 兼容生成模型 API 地址"
        )
    else:
        base_url = configured_base_url

    timeout = _env_float("LLM_REQUEST_TIMEOUT_SECONDS", 180, minimum=1)
    max_retries = _env_int("LLM_MAX_RETRIES", 1, minimum=0)
    extra_body = None
    if provider == "dashscope":
        extra_body = {"enable_thinking": _env_bool("LLM_ENABLE_THINKING", False)}

    client = create_openai_client(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    return OpenAICompatibleChatModel(
        client=client,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        native_json_mode=_json_mode(provider),
        token_limit_parameter=_token_limit_parameter(),
        extra_body=extra_body,
    )


def get_simulation_llm():
    return SimulationLLMAdapter(get_generation_chat_model(temperature=0.65))


def _embedding_provider() -> str:
    # Embeddings deliberately do not inherit the chat provider: the two services
    # can use different vendors, credentials and base URLs.
    provider = _normalized_provider(os.getenv("EMBEDDING_PROVIDER") or "dashscope")
    if provider not in EMBEDDING_PROVIDERS:
        raise unsupported_provider_error(provider, "向量模型")
    return provider


def _embedding_batch_size(provider: str) -> int:
    is_dashscope_native = provider in {"dashscope", "dashscope_native"}
    if is_dashscope_native:
        configured = _env_int(
            "EMBEDDING_BATCH_SIZE",
            DASHSCOPE_MAX_EMBEDDING_BATCH_SIZE,
            minimum=1,
        )
        return min(configured, DASHSCOPE_MAX_EMBEDDING_BATCH_SIZE)
    default = (
        OPENAI_DEFAULT_EMBEDDING_BATCH_SIZE
        if provider == "openai"
        else COMPATIBLE_DEFAULT_EMBEDDING_BATCH_SIZE
    )
    return min(_env_int("EMBEDDING_BATCH_SIZE", default, minimum=1), 2048)


def get_embedding_model():
    provider = _embedding_provider()
    model_name = _required_env("EMBEDDING_MODEL", "向量模型名称")
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise configuration_error("EMBEDDING_API_KEY", "向量模型 API Key")
    batch_size = _embedding_batch_size(provider)

    if provider in {"dashscope", "dashscope_native"}:
        from llama_index.embeddings.dashscope import DashScopeEmbedding

        return DashScopeEmbedding(
            model_name=model_name,
            api_key=api_key,
            embed_batch_size=batch_size,
        )

    configured_base_url = (os.getenv("EMBEDDING_API_BASE") or "").strip() or None
    if provider == "dashscope_compatible":
        base_url = configured_base_url or DASHSCOPE_COMPATIBLE_BASE_URL
    elif provider == "openai_compatible":
        base_url = configured_base_url or _required_env(
            "EMBEDDING_API_BASE", "OpenAI 兼容向量模型 API 地址"
        )
    else:
        base_url = configured_base_url

    from .embeddings import OpenAICompatibleEmbedding

    client = create_openai_client(
        api_key=api_key,
        base_url=base_url,
        timeout=_env_float("EMBEDDING_REQUEST_TIMEOUT_SECONDS", 180, minimum=1),
        max_retries=_env_int("EMBEDDING_MAX_RETRIES", 1, minimum=0),
    )
    return OpenAICompatibleEmbedding(
        client=client,
        model_name=model_name,
        embed_batch_size=batch_size,
    )
