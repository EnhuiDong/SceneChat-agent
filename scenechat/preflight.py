from dataclasses import dataclass

from .errors import SceneChatError, classify_provider_error
from .openai_compat import response_content
from .providers import get_embedding_model, get_generation_chat_model
from .scenario import extract_json_object
from .build_control import CURRENT_BUILD, BuildControl
from .telemetry import measured_call
from .recovery import transport_call
from .config import config_int, validate_simulation_modes


def _probe_request(operation):
    control = CURRENT_BUILD.get()
    if control is None:
        with BuildControl().activate():
            return _probe_request(operation)
    return transport_call(operation, control=control,
                          retries=config_int("scenario", "transport_retries", 1, minimum=0, maximum=2),
                          limit=config_int("scenario", "max_requests_per_step", 4, minimum=1, maximum=8))


@dataclass(frozen=True)
class ModelPreflightResult:
    generation_model: str
    embedding_model: str


def _probe_generation_model(generation_model) -> None:
    control = CURRENT_BUILD.get()
    if control is None:
        with BuildControl().activate():
            return _probe_generation_model(generation_model)
    repairs = min(1, config_int("scenario", "json_repair_retries", 2, minimum=0, maximum=2))
    for attempt in range(repairs + 1):
        response = _probe_request(lambda: measured_call(
            control.model_requests, stage="preflight", purpose="chat_availability",
            model=getattr(generation_model, "model_name", "configured"),
            operation=lambda: generation_model.invoke(
                [("user", '只输出这个 JSON 对象：{"ok":true}' + (
                    "。上次输出结构无效，不要解释或使用代码块。" if attempt else ""))],
                max_tokens=24 if attempt else 12,
                response_format={"type": "json_object"},
            ),
        ))
        try:
            payload = extract_json_object(response_content(response))
            if payload.get("ok") is not True:
                raise ValueError("生成模型没有按要求返回 JSON")
            return
        except (ValueError, TypeError):
            if attempt >= repairs:
                raise


def validate_model_availability() -> ModelPreflightResult:
    """Fail fast with minimal requests before expensive document generation."""
    # Construct both clients first so missing configuration is reported without
    # making any external request.
    validate_simulation_modes()
    embedding_model = get_embedding_model()
    generation_model = get_generation_chat_model(temperature=0)

    try:
        # The batch API is the same one used by real indexing. Embeddings must
        # pass before spending tokens on chat, including in the CLI path.
        try:
            _probe_request(lambda: embedding_model.get_text_embedding_batch(
                ["SceneChat 模型可用性检查 A", "SceneChat 模型可用性检查 B"]
            ))
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
    finally:
        close = getattr(generation_model, "close", None)
        if close:
            close()

    return ModelPreflightResult(
        generation_model=str(generation_model.model_name),
        embedding_model=str(getattr(embedding_model, "model_name", "configured")),
    )


def validate_generation_model_availability() -> str:
    """Probe the same JSON path used by scenario and simulation generation."""
    validate_simulation_modes()
    generation_model = get_generation_chat_model(temperature=0, max_tokens=12)
    try:
        _probe_generation_model(generation_model)
    except SceneChatError:
        raise
    except Exception as exc:
        raise classify_provider_error(exc, service="生成模型") from exc
    finally:
        close = getattr(generation_model, "close", None)
        if close:
            close()
    return str(generation_model.model_name)


def validate_embedding_model_availability() -> str:
    """Probe the indexing batch API before spending tokens on scenario generation."""
    validate_simulation_modes()
    embedding_model = get_embedding_model()
    try:
        _probe_request(lambda: embedding_model.get_text_embedding_batch(
            ["SceneChat 向量检查 A", "SceneChat 向量检查 B"]
        ))
    except SceneChatError:
        raise
    except Exception as exc:
        raise classify_provider_error(exc, service="向量模型") from exc
    return str(getattr(embedding_model, "model_name", "configured"))
