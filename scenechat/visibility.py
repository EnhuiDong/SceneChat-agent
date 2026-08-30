from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PUBLIC = "public"
DIRECTOR_ONLY = "director_only"
AUDIENCE_ONLY = "audience_only"


def normalize_scopes(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return [PUBLIC]
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    else:
        raw = [str(part).strip() for part in value]
    scopes = [scope for scope in raw if scope]
    return list(dict.fromkeys(scopes)) or [PUBLIC]


@dataclass(frozen=True)
class ViewerContext:
    name: str = ""
    role: str = ""
    location: str = ""
    is_director: bool = False
    is_audience: bool = False


def can_access(scopes: str | Iterable[str] | None, viewer: ViewerContext) -> bool:
    """Return whether a viewer may receive information under any given scope.

    Director access is intentionally explicit. Audience access permits reader
    camera information but not arbitrary per-agent secrets; director narration
    can request the complete corpus separately.
    """

    normalized = normalize_scopes(scopes)
    if viewer.is_director:
        return True
    for scope in normalized:
        if scope == PUBLIC:
            return True
        if scope == AUDIENCE_ONLY and viewer.is_audience:
            return True
        if scope == DIRECTOR_ONLY:
            continue
        prefix, separator, value = scope.partition(":")
        if not separator:
            continue
        value = value.strip()
        if prefix == "agent" and value == viewer.name:
            return True
        if prefix == "role" and value == viewer.role:
            return True
        if prefix == "location" and value == viewer.location:
            return True
    return False


def access_keys_for(viewer: ViewerContext) -> list[str]:
    keys = [PUBLIC]
    if viewer.name:
        keys.append(f"agent:{viewer.name}")
    if viewer.role:
        keys.append(f"role:{viewer.role}")
    if viewer.location:
        keys.append(f"location:{viewer.location}")
    if viewer.is_audience:
        keys.append(AUDIENCE_ONLY)
    if viewer.is_director:
        keys.append("director")
    return list(dict.fromkeys(keys))
