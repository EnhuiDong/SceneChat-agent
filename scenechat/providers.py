from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from .config import config_value
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
            getattr(prompt, "messages", None) or [("user", prompt)],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = getattr(response, "raw", None)
        choices = getattr(raw, "choices", None) or []
        finish_reason = getattr(choices[0], "finish_reason", "") if choices else ""
        return SimpleNamespace(
            text=str(getattr(response, "content", "") or ""),
            raw=raw,
            finish_reason=str(finish_reason or ""),
        )


def _normalized_provider(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _required_env(name: str, description: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise configuration_error(name, description)
    return value


def _invalid_setting(name: str, description: str) -> SceneChatError:
    return SceneChatError(
        "model_configuration_invalid",
        f"后端配置的{description}（{name}）无效，请完善 config.json 后重新启动服务。",
        stage="preflight",
        status_code=503,
    )


def _setting_bool(section: str, key: str, default: bool = False) -> bool:
    raw = config_value(section, key, default)
    if isinstance(raw, bool):
        return raw
    if raw is None or not str(raw).strip():
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise _invalid_setting(f"{section}.{key}", "布尔值")


def _setting_int(
    section: str, key: str, default: int, *, minimum: int = 0
) -> int:
    raw = config_value(section, key, default)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise _invalid_setting(f"{section}.{key}", "整数") from exc
    if value < minimum:
        raise _invalid_setting(f"{section}.{key}", f"不小于 {minimum} 的整数")
    return value


def _setting_float(
    section: str, key: str, default: float, *, minimum: float = 0
) -> float:
    raw = config_value(section, key, default)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise _invalid_setting(f"{section}.{key}", "数字") from exc
    if value < minimum:
        raise _invalid_setting(f"{section}.{key}", f"不小于 {minimum} 的数字")
    return value


def _chat_provider() -> str:
    provider = _normalized_provider(
        os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or "dashscope"
    )
    if provider not in CHAT_PROVIDERS:
        raise unsupported_provider_error(provider, "生成模型")
    return provider


def _json_mode(provider: str) -> bool:
    raw = str(config_value("llm", "json_mode", "auto")).strip().lower()
    if raw == "auto":
        return provider in {"dashscope", "openai"}
    if raw in {"native", "on", "true", "1"}:
        return True
    if raw in {"prompt", "off", "false", "0"}:
        return False
    raise _invalid_setting("llm.json_mode", "JSON 模式")


def _token_limit_parameter() -> str:
    value = str(config_value("llm", "token_limit_parameter", "max_tokens")).strip()
    if value not in {"max_tokens", "max_completion_tokens"}:
        raise _invalid_setting("llm.token_limit_parameter", "输出 token 参数名")
    return value


def get_generation_chat_model(
    temperature: float = 0.8,
    max_tokens: int | None = None,
    *,
    transport_retries: int | None = None,
):
    provider = _chat_provider()
    api_key = _required_env("LLM_API_KEY", "生成模型 API Key")
    model_name = _required_env("LLM_MODEL", "生成模型名称")

    configured_base_url = (os.getenv("LLM_API_BASE") or "").strip() or None
    if provider == "dashscope":
        base_url = configured_base_url or DASHSCOPE_COMPATIBLE_BASE_URL
    elif provider == "openai_compatible":
        base_url = configured_base_url
        if not base_url:
            raise configuration_error("LLM_API_BASE", "OpenAI 兼容生成模型 API 地址")
    else:
        base_url = configured_base_url

    timeout = _setting_float("llm", "request_timeout_seconds", 180, minimum=1)
    max_retries = _setting_int("llm", "max_retries", 1, minimum=0)
    if transport_retries is not None:
        max_retries = transport_retries
    extra_body = None
    if provider == "dashscope":
        extra_body = {
            "enable_thinking": _setting_bool("llm", "enable_thinking", False)
        }

    # Capability differences belong to public configuration, not to credentials
    # or implicit model-name heuristics. The selected model still comes from env.
    profiles = config_value("llm", "model_profiles", {})
    if not isinstance(profiles, dict):
        raise _invalid_setting("llm.model_profiles", "模型参数配置")
    provider_profiles = profiles.get(provider, {})
    if not isinstance(provider_profiles, dict):
        raise _invalid_setting("llm.model_profiles", "供应商参数配置")
    profile = provider_profiles.get(model_name, {})
    allowed = {"temperature", "enable_thinking", "json_mode", "reasoning_token_reserve"}
    if not isinstance(profile, dict) or set(profile) - allowed:
        raise _invalid_setting("llm.model_profiles", "模型参数字段")
    native_json_mode = _json_mode(provider)
    reasoning_reserve = profile.get("reasoning_token_reserve", 0)
    if (isinstance(reasoning_reserve, bool) or not isinstance(reasoning_reserve, int)
            or not 0 <= reasoning_reserve <= 32768):
        raise _invalid_setting("llm.model_profiles.reasoning_token_reserve", "推理 token 预留")
    if "temperature" in profile:
        temperature = profile["temperature"]
        if (temperature is not None and (isinstance(temperature, bool)
                or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2)):
            raise _invalid_setting("llm.model_profiles.temperature", "温度参数")
    if "enable_thinking" in profile:
        if not isinstance(profile["enable_thinking"], bool) or provider != "dashscope":
            raise _invalid_setting("llm.model_profiles.enable_thinking", "思考模式参数")
        extra_body["enable_thinking"] = profile["enable_thinking"]
    if "json_mode" in profile:
        if not isinstance(profile["json_mode"], str) or profile["json_mode"] not in {"native", "prompt"}:
            raise _invalid_setting("llm.model_profiles.json_mode", "模型 JSON 模式")
        native_json_mode = profile["json_mode"] == "native"

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
        native_json_mode=native_json_mode,
        token_limit_parameter=_token_limit_parameter(),
        extra_body=extra_body,
        reasoning_token_reserve=reasoning_reserve,
    )


def get_simulation_llm():
    return SimulationLLMAdapter(get_generation_chat_model(temperature=0.65))


def _embedding_provider() -> str:
    # Embeddings deliberately do not inherit the chat provider: the two services
    # can use different vendors, credentials and base URLs.
    provider = _normalized_provider(
        os.getenv("EMBEDDING_PROVIDER") or "dashscope"
    )
    if provider not in EMBEDDING_PROVIDERS:
        raise unsupported_provider_error(provider, "向量模型")
    return provider


def _embedding_batch_size(provider: str) -> int:
    is_dashscope_native = provider in {"dashscope", "dashscope_native"}
    if is_dashscope_native:
        configured = _setting_int(
            "embedding",
            "batch_size",
            DASHSCOPE_MAX_EMBEDDING_BATCH_SIZE,
            minimum=1,
        )
        return min(configured, DASHSCOPE_MAX_EMBEDDING_BATCH_SIZE)
    default = (
        OPENAI_DEFAULT_EMBEDDING_BATCH_SIZE
        if provider == "openai"
        else COMPATIBLE_DEFAULT_EMBEDDING_BATCH_SIZE
    )
    return min(_setting_int("embedding", "batch_size", default, minimum=1), 2048)


def get_embedding_model():
    provider = _embedding_provider()
    model_name = _required_env("EMBEDDING_MODEL", "向量模型名称")
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise configuration_error("EMBEDDING_API_KEY", "向量模型 API Key")
    batch_size = _embedding_batch_size(provider)

    if provider in {"dashscope", "dashscope_native"}:
        from .dashscope_embedding import CheckedDashScopeEmbedding

        return CheckedDashScopeEmbedding(
            model_name=model_name,
            api_key=api_key,
            embed_batch_size=batch_size,
        )

    configured_base_url = (os.getenv("EMBEDDING_API_BASE") or "").strip() or None
    if provider == "dashscope_compatible":
        base_url = configured_base_url or DASHSCOPE_COMPATIBLE_BASE_URL
    elif provider == "openai_compatible":
        base_url = configured_base_url
        if not base_url:
            raise configuration_error("EMBEDDING_API_BASE", "OpenAI 兼容向量模型 API 地址")
    else:
        base_url = configured_base_url

    from .embeddings import OpenAICompatibleEmbedding

    client = create_openai_client(
        api_key=api_key,
        base_url=base_url,
        timeout=_setting_float(
            "embedding", "request_timeout_seconds", 180, minimum=1
        ),
        max_retries=_setting_int(
            "embedding", "max_retries", 1, minimum=0
        ),
    )
    return OpenAICompatibleEmbedding(
        client=client,
        model_name=model_name,
        embed_batch_size=batch_size,
    )
