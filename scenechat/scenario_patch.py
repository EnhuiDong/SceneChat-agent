"""Apply a bounded, atomic field patch without regenerating valid documents."""

from copy import deepcopy
from dataclasses import fields

from .scenario import CharacterSpec, ScenarioPackage, ScenarioValidationError, WorldSpec


def apply_scenario_patch(package, payload):
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
        if not valid or any(path == prior or path.startswith(prior + "/") or prior.startswith(path + "/") for prior in seen):
            raise ScenarioValidationError(["补丁包含越界、重叠或禁止修改的路径；约束账本、人数及角色 ID 不可修改"])
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
            if previous is not None and type(previous) is not type(value):
                if not (type(previous) is float and type(value) is int):
                    raise TypeError
            node[key] = deepcopy(change["value"])
        except (KeyError, ValueError, IndexError, TypeError):
            raise ScenarioValidationError(["补丁路径不存在；请替换已声明字段，不得构造虚假层级"]) from None
    if data == package.to_dict():
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
