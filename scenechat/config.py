from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load the public, non-secret application configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("config.json 顶层必须是 JSON 对象")
    return payload


def config_value(section: str, key: str, default: Any = None) -> Any:
    values = load_config().get(section)
    if not isinstance(values, dict):
        return default
    return values.get(key, default)


def config_int(
    section: str,
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(config_value(section, key, default))
    except (TypeError, ValueError):
        return default
    return value if minimum <= value <= maximum else default
