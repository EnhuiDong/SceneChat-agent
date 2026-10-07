"""Apply a bounded, atomic field patch without regenerating valid documents."""

from copy import deepcopy
from dataclasses import fields
import json
import re

from .scenario import CharacterSpec, ScenarioPackage, ScenarioValidationError, WorldSpec


def _polymorphic_value(parts):
    """Only protocol fields declared Any may legitimately change JSON type."""
    path = "/" + "/".join(parts)
    return bool(re.fullmatch(
        r"/(?:world/rules/\d+/effects/\d+|characters/\d+/abilities/\d+/effects/\d+"
        r"|world/termination_rules/\d+(?:/conditions/\d+)*|world/rules/\d+/requires/\d+"
        r"|world/beat_specs/\d+/completion_conditions/\d+)/value", path))


def apply_scenario_patch(package, payload, *, allow_noop=False):
    # A well-formed no-op is a semantic non-progress result, not broken JSON.
    # Only generation opts in so its semantic ledger accounts for the attempt;
    # interactive callers still reject it. Path/type/identity guards never relax.
    changes = payload.get("changes")
    if not isinstance(changes, list) or not 1 <= len(changes) <= 80:
        raise ScenarioValidationError(["修复响应必须包含 1–80 项 changes 字段补丁，不能返回整份设定"])
    data = deepcopy(package.to_dict())
    seen = set()
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"path", "value"}:
            raise ScenarioValidationError(["每项补丁必须只包含 path 和 value"])
        path = change["path"]
        if not isinstance(path, str) or not path.startswith("/"):
            raise ScenarioValidationError(["补丁 path 必须是绝对 JSON Pointer"])
        parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
        if parts[:2] == ["world", "execution_version"]:
            raise ScenarioValidationError(["自动修复不得修改 execution_version 或降级规则语义"])
        valid = parts[0] == "world" and len(parts) >= 2 and parts[1] in data["world"]
        if parts[0] == "characters" and len(parts) >= 3:
            valid = parts[1].isdigit() and int(parts[1]) < len(data["characters"])
            valid = valid and parts[2] in data["characters"][int(parts[1])] and parts[2] != "id"
        if not valid:
            raise ScenarioValidationError([f"补丁包含越界或禁止修改的路径 {path}；约束账本、人数及角色 ID 不可修改"])
        conflict = next((prior for prior in sorted(seen)
                         if path == prior or path.startswith(prior + "/") or prior.startswith(path + "/")), None)
        if conflict:
            raise ScenarioValidationError([
                f"补丁路径重叠：{path} 与 {conflict}；替换父字段时不要同时补丁它的子字段，"
                "把必要子字段修改合并到父字段 value，或只保留不重叠的子字段补丁"
            ])
        seen.add(path)
        node = data
        try:
            for part in parts[:-1]:
                node = node[int(part)] if isinstance(node, list) else node[part]
            key = int(parts[-1]) if isinstance(node, list) else parts[-1]
            if isinstance(node, list) and (key < 0 or key >= len(node)):
                raise IndexError
            if not isinstance(node, (dict, list)):
                raise TypeError
            previous = node[key] if isinstance(node, list) or key in node else None
            value = change["value"]
            if previous is not None and type(previous) is not type(value) and not _polymorphic_value(parts):
                if not (type(previous) is float and type(value) is int):
                    raise TypeError
            node[key] = deepcopy(change["value"])
        except (KeyError, ValueError, IndexError, TypeError):
            raise ScenarioValidationError([f"补丁路径 {path} 不存在或 value 类型不匹配；请替换已声明字段，不得构造虚假层级"]) from None
    # Python equality conflates True and 1 inside otherwise identical objects.
    # JSON type changes in Any-valued fields are real semantic modifications.
    if json.dumps(data, sort_keys=True) == json.dumps(package.to_dict(), sort_keys=True) and not allow_noop:
        raise ScenarioValidationError(["修复补丁未产生任何变化，停止重复修复"])
    result = ScenarioPackage(
        brief=package.brief, world=WorldSpec.from_mapping(data["world"]),
        characters=[CharacterSpec.from_mapping(item, i) for i, item in enumerate(data["characters"], 1)],
        warnings=list(package.warnings),
    )
    # Mapping constructors intentionally fill defaults for newly generated
    # documents. A repair must not apply those migrations to untouched fields.
    touched_world = {path.split("/")[2] for path in seen if path.startswith("/world/")}
    for field in fields(package.world):
        if field.name not in touched_world:
            setattr(result.world, field.name, deepcopy(getattr(package.world, field.name)))
    for index, actor in enumerate(package.characters):
        prefix = f"/characters/{index}/"
        touched = {path[len(prefix):].split("/")[0] for path in seen if path.startswith(prefix)}
        for field in fields(actor):
            if field.name not in touched:
                setattr(result.characters[index], field.name, deepcopy(getattr(actor, field.name)))
    return result
