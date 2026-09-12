"""Validate user-owned imports without executing or trusting their contents."""
import json


def validate_import(payload):
    from .persistence import SCHEMA_VERSION, runtime_session_from_export
    if not isinstance(payload, dict):
        raise ValueError("请选择 SceneChat JSON 完整导出文件。")
    version = payload.get("schema_version", 1)
    if type(version) is not int or not 1 <= version <= SCHEMA_VERSION:
        raise ValueError("不支持此存档版本，请先更新程序。")
    try:
        serialized_size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("存档含无效数值或嵌套结构。") from exc
    if serialized_size > 20 * 1024 * 1024:
        raise ValueError("存档超过 20 MB 导入上限。")
    for key in ("session", "scenario", "simulation"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"存档缺少 {key} 完整状态。")
    sim = payload["simulation"]
    agents, history = sim.get("agents"), sim.get("history")
    if not isinstance(agents, list) or not agents or not isinstance(history, list):
        raise ValueError("存档缺少角色或事件状态。")
    names = [a.get("name") for a in agents if isinstance(a, dict)]
    ids = [m.get("event_id") for m in history if isinstance(m, dict)]
    if len(names) != len(agents) or not all(isinstance(n, str) and n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("角色名称缺失或重复。")
    if len(ids) != len(history) or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("事件标识缺失或重复。")
    if sim.get("turn_count", len(history)) != len(history):
        raise ValueError("事件计数与完整历史不一致。")
    for key in ("turn_count", "revision"):
        if type(sim.get(key, 0)) is not int or sim.get(key, 0) < 0:
            raise ValueError("存档计数或版本无效。")
    cast = payload["scenario"].get("characters")
    if not isinstance(cast, list) or any(not isinstance(c, dict) or not isinstance(c.get("name"), str) for c in cast):
        raise ValueError("存档缺少有效的完整人物设定。")
    if len(cast) != len(names) or {c["name"] for c in cast} != set(names):
        raise ValueError("人物设定与运行状态名单不一致。")
    for agent in agents:
        for key in ("profile", "public_profile", "name"):
            if not isinstance(agent.get(key), str):
                raise ValueError("角色档案字段必须是文本。")
        for key in ("memories", "memory_archive", "belief_records", "pending_commitments", "memory_summaries"):
            if key in agent and not isinstance(agent[key], list):
                raise ValueError("角色记忆字段格式无效。")
    for event in history:
        if any(not isinstance(event.get(key), str) for key in ("speaker", "action", "speech")):
            raise ValueError("事件内容字段必须是文本。")
        if not isinstance(event.get("intent", {}), dict) or not isinstance(event.get("state_patch", []), list):
            raise ValueError("事件意图或状态操作格式无效。")
        if type(event.get("turn")) is not int or event["turn"] < 0:
            raise ValueError("事件轮次无效。")
    try:
        restored = runtime_session_from_export(payload)
        from .mechanics import validate_mechanics
        if validate_mechanics(restored["scenario"]):
            raise ValueError("存档包含当前程序不支持的执行规则。")
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ValueError("存档字段格式无效，无法安全恢复。") from exc
    return {"title": restored["scenario"].world.title, "characters": len(names),
            "turns": len(history), "schema_version": version, "migrated_to": SCHEMA_VERSION,
            "warning": "完整存档含人物秘密。将创建独立推演，不覆盖已有数据；只恢复文件中的完整状态。"}
