from __future__ import annotations

from typing import Any


class SceneChatError(Exception):
    """An internal error with a stable, user-safe API representation."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        stage: str,
        status_code: int = 503,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.stage = stage
        self.status_code = status_code
        self.cause = cause

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.public_message,
                "stage": self.stage,
            }
        }


def configuration_error(variable: str, description: str) -> SceneChatError:
    return SceneChatError(
        "model_configuration_missing",
        f"后端尚未配置{description}（{variable}），请完善 .env 后重新启动服务。",
        stage="preflight",
        status_code=503,
    )


def unsupported_provider_error(provider: str, service: str) -> SceneChatError:
    return SceneChatError(
        "model_provider_unsupported",
        f"当前后端暂不支持配置的{service}提供商“{provider}”，请检查 .env。",
        stage="preflight",
        status_code=503,
    )


def classify_provider_error(
    exc: Exception,
    *,
    service: str,
    stage: str = "preflight",
) -> SceneChatError:
    """Translate provider-specific failures without exposing raw responses."""
    text = str(exc).lower()
    service_code = "embedding" if service == "向量模型" else "generation_model"

    quota_markers = (
        "allocationquota",
        "free quota exhausted",
        "quota exhausted",
        "insufficient balance",
        "insufficient_quota",
        "余额不足",
        "额度已用尽",
    )
    auth_markers = (
        "invalidapikey",
        "invalid api key",
        "invalid_api_key",
        "unauthorized",
        "authentication",
        "access denied",
        "401",
    )
    model_markers = (
        "model not found",
        "model_not_found",
        "invalid model",
        "does not exist",
        "模型不存在",
    )
    network_markers = (
        "connection",
        "timeout",
        "timed out",
        "network",
        "dns",
    )

    if any(marker in text for marker in quota_markers):
        return SceneChatError(
            f"{service_code}_quota_exhausted",
            f"{service}额度不足或免费额度已用尽，请在模型服务控制台补充额度后重试。",
            stage=stage,
            cause=exc,
        )
    if any(marker in text for marker in auth_markers):
        return SceneChatError(
            f"{service_code}_authentication_failed",
            f"{service}认证失败，请检查 API Key 是否正确且具有该模型的访问权限。",
            stage=stage,
            cause=exc,
        )
    if any(marker in text for marker in model_markers):
        return SceneChatError(
            f"{service_code}_not_found",
            f"配置的{service}不存在或当前账户无权使用，请检查模型名称和访问权限。",
            stage=stage,
            cause=exc,
        )
    if any(marker in text for marker in network_markers):
        return SceneChatError(
            f"{service_code}_connection_failed",
            f"暂时无法连接{service}服务，请检查网络或稍后重试。",
            stage=stage,
            cause=exc,
        )
    return SceneChatError(
        f"{service_code}_unavailable",
        f"{service}当前不可用，请检查服务配置或稍后重试。",
        stage=stage,
        cause=exc,
    )


STAGE_MESSAGES = {
    "scenario_generation": (
        "scenario_generation_failed",
        "场景设定生成或一致性校验失败，请检查输入后重试；系统不会启动残缺的模拟。",
    ),
    "worldview": (
        "worldview_generation_failed",
        "世界观生成失败，请稍后重试；若问题持续，请检查生成模型配置。",
    ),
    "characters": (
        "character_generation_failed",
        "角色生成失败，请稍后重试；已生成的世界观不会作为完整实验启动。",
    ),
    "character_parsing": (
        "character_parsing_failed",
        "角色档案格式不完整，暂时无法开始实验，请重新生成。",
    ),
    "index": (
        "knowledge_index_failed",
        "实验知识库创建失败，请检查向量模型状态后重试。",
    ),
    "storage": (
        "experiment_storage_failed",
        "实验内容已生成，但保存失败，请检查后端数据目录权限。",
    ),
    "simulation": (
        "simulation_generation_failed",
        "故事续写失败，请稍后重试当前页面。",
    ),
    "intervention": (
        "intervention_preview_failed",
        "剧情干预预检失败，请稍后重试；现有剧情状态没有被修改。",
    ),
    "simulation_client": (
        "simulation_client_failed",
        "推演模型初始化失败，请检查模型配置后重试。",
    ),
}


def stage_error(stage: str, exc: Exception) -> SceneChatError:
    if isinstance(exc, SceneChatError):
        return exc
    code, message = STAGE_MESSAGES.get(
        stage,
        ("internal_error", "服务暂时无法完成请求，请稍后重试。"),
    )
    return SceneChatError(code, message, stage=stage, cause=exc)
