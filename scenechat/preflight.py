from dataclasses import dataclass

from .errors import SceneChatError, classify_provider_error
from .openai_compat import response_content
from .providers import get_embedding_model, get_generation_chat_model
from .scenario import extract_json_object


@dataclass(frozen=True)
class ModelPreflightResult:
    generation_model: str
    embedding_model: str


def _probe_generation_model(generation_model) -> None:
    response = generation_model.invoke(
        [("user", '只输出这个 JSON 对象：{"ok":true}')],
        max_tokens=12,
        response_format={"type": "json_object"},
    )
    payload = extract_json_object(response_content(response))
    if payload.get("ok") is not True:
        raise ValueError("生成模型没有按要求返回 JSON")


def validate_model_availability() -> ModelPreflightResult:
    """Fail fast with minimal requests before expensive document generation."""
    # Construct both clients first so missing configuration is reported without
    # making any external request.
    embedding_model = get_embedding_model()
    generation_model = get_generation_chat_model(temperature=0)

    # Check embedding first: it is required to finish /start and is cheaper to
    # probe than generating the world and character documents.
    try:
        # Use the batch API used during actual indexing, even though the probe
        # contains only two cheap inputs.
        embedding_model.get_text_embedding_batch(
            ["SceneChat 模型可用性检查 A", "SceneChat 模型可用性检查 B"]
        )
    except SceneChatError:
        raise
    except Exception as exc:
        raise classify_provider_error(exc, service="向量模型") from exc

    try:
        _probe_generation_model(generation_model)
    except SceneChatError:
        raise
    except Exception as exc:
        raise classify_provider_error(exc, service="生成模型") from exc

    return ModelPreflightResult(
        generation_model=str(generation_model.model_name),
        embedding_model=str(getattr(embedding_model, "model_name", "configured")),
    )


def validate_generation_model_availability() -> str:
    """Probe the same JSON path used by scenario and simulation generation."""
    generation_model = get_generation_chat_model(temperature=0, max_tokens=12)
    try:
        _probe_generation_model(generation_model)
    except SceneChatError:
        raise
    except Exception as exc:
        raise classify_provider_error(exc, service="生成模型") from exc
    return str(generation_model.model_name)


def validate_embedding_model_availability() -> str:
    """Probe Embedding only when the generated background requires vector RAG."""
    embedding_model = get_embedding_model()
    try:
        embedding_model.get_text_embedding_batch(
            ["SceneChat 向量检查 A", "SceneChat 向量检查 B"]
        )
    except SceneChatError:
        raise
    except Exception as exc:
        raise classify_provider_error(exc, service="向量模型") from exc
    return str(getattr(embedding_model, "model_name", "configured"))
