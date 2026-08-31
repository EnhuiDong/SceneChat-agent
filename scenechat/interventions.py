from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from .context import director_context
from .models import Intervention, Message, SimulationState


INTERVENTION_PROMPT = """你是互动故事的导演指令分析器。用户希望干预已经开始的剧情。

把输入分类为：
- guidance：只影响后续倾向，不宣布事实、不直接改状态；
- event：在下一步发生一个可观察事件，可更新已声明的公共状态；
- override：明确改写既有事实、角色状态、阶段或硬设定。只有用户确实要求强行改写时才选。

不得替用户扩大意图。不得把秘密注入不知情角色的头脑。不得输出任意代码或未列出的 patch 操作。
识别与已发生事件、固定设定、世界规则或角色自主性的冲突。冲突 severity 只能是 warning 或 blocking。
符合题材的新天气、道具、路人、来信、车辆等环境事件可以自然进入故事，不因“此前未出现”而算冲突；只有冒充既有角色/地点/状态字段、否定已发生事实或越过规则时才属于 unknown_reference/continuity 冲突。

只输出 JSON：
{
  "mode":"guidance|event|override",
  "normalized_directive":"忠实、简洁、可执行的导演指令",
  "event_narration":"event/override 时实际进入时间线的一至三句客观事件；guidance 为空",
  "visibility":"public|audience_only",
  "proposed_patch":[{"op":"set_world|increment_world|move_agent|set_agent_status|set_resource|set_goal_status|set_relationship|set_phase|add_known_fact","target":"角色名","key":"字段","value":"值","amount":1}],
  "conflicts":[{"severity":"warning|blocking","kind":"canon|rule|continuity|agency|unknown_reference","message":"给用户看的具体说明"}]
}
"""

EVENT_OPS = {"set_world", "increment_world", "move_agent"}
OVERRIDE_OPS = EVENT_OPS | {
    "set_agent_status", "set_resource", "set_goal_status", "set_relationship",
    "set_phase", "add_known_fact",
}


def _extract_json(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", value, re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("干预分析结果不是 JSON 对象")
    return payload


def _conflict(severity: str, kind: str, message: str) -> dict[str, str]:
    return {"severity": severity, "kind": kind, "message": message}


def validate_intervention(state: SimulationState, intervention: Intervention) -> Intervention:
    """Apply deterministic reference and patch validation after model analysis."""

    conflicts = [
        item for item in intervention.conflicts
        if item.get("severity") in {"warning", "blocking"} and str(item.get("message") or "").strip()
    ][:12]
    if intervention.mode == "event":
        for item in conflicts:
            if item.get("kind") == "unknown_reference" and item.get("severity") == "blocking":
                item["severity"] = "warning"
                item["message"] = f"{item['message']}；将作为新环境元素处理，不会据此越权修改结构化状态。"
    allowed = OVERRIDE_OPS if intervention.mode == "override" else EVENT_OPS
    safe_patch: list[dict[str, Any]] = []
    proposed_operations = [] if intervention.mode == "guidance" else intervention.proposed_patch[:30]
    for raw in proposed_operations:
        operation = dict(raw)
        op = str(operation.get("op") or "").strip()
        target = str(operation.get("target") or "").strip()
        key = str(operation.get("key") or "").strip()
        value = operation.get("value")
        reason = ""
        if op not in allowed:
            reason = f"操作 {op or '（空）'} 不允许用于 {intervention.mode} 干预"
        elif op in {"move_agent", "set_agent_status", "set_resource", "set_goal_status", "add_known_fact"} and target not in state.agents:
            reason = f"引用了不存在的角色“{target}”"
        elif op == "set_relationship" and (target not in state.agents or key not in state.agents):
            reason = "关系更新引用了不存在的角色"
        elif op in {"set_resource", "set_goal_status", "add_known_fact"} and not key:
            reason = f"操作 {op} 缺少字段名"
        elif op == "move_agent" and state.locations and str(value) not in state.locations:
            reason = f"引用了不存在的地点“{value}”"
        elif op == "move_agent" and not str(value or "").strip():
            reason = "移动操作缺少目标地点"
        elif op == "set_phase" and str(value) not in state.phase_sequence:
            reason = f"引用了不存在的阶段“{value}”"
        elif op in {"set_world", "increment_world"} and state.state_schema and key not in state.state_schema:
            reason = f"引用了未声明的世界状态“{key}”"
        elif op == "set_world" and not state._valid_world_value(key, value):
            reason = f"世界状态“{key}”的值不符合 schema"
        elif op == "increment_world" and (
            not key
            or not isinstance(operation.get("amount", 1), int)
            or isinstance(operation.get("amount", 1), bool)
            or not isinstance(state.world_state.get(key, 0), int)
            or isinstance(state.world_state.get(key, 0), bool)
        ):
            reason = f"世界状态“{key}”不是可安全递增的整数"
        elif op == "set_agent_status" and key not in {"active", "alive"}:
            reason = "角色状态只能修改 active 或 alive"
        elif op == "set_agent_status" and not isinstance(value, bool):
            reason = "角色状态值必须是 true 或 false"
        if reason:
            conflicts.append(_conflict("blocking", "unknown_reference", reason))
        else:
            safe_patch.append(operation)

    if intervention.mode == "guidance":
        safe_patch = []
        intervention.event_narration = ""
    elif not intervention.event_narration:
        conflicts.append(_conflict("blocking", "continuity", "事件型干预缺少可进入时间线的事件描述"))
    if intervention.visibility not in {"public", "audience_only"}:
        intervention.visibility = "public"
    if intervention.visibility == "audience_only" and safe_patch:
        conflicts.append(_conflict("blocking", "rule", "仅观众可见的事件不能直接修改公共状态"))
        safe_patch = []

    # Deduplicate deterministic/model conflicts without hiding independent issues.
    seen = set()
    intervention.conflicts = []
    for item in conflicts:
        signature = (item.get("severity"), item.get("kind"), item.get("message"))
        if signature not in seen:
            seen.add(signature)
            intervention.conflicts.append(item)
    intervention.proposed_patch = safe_patch
    return intervention


def preview_intervention(
    state: SimulationState,
    raw_text: str,
    *,
    scope: str = "next_scene",
    expires_after_turns: int | None = None,
    llm=None,
) -> Intervention:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("干预内容不能为空")
    if len(raw_text) > 3000:
        raise ValueError("单次干预不能超过 3000 个字符")
    if llm is None:
        from .providers import get_simulation_llm

        llm = get_simulation_llm()
    user_prompt = (
        f"【当前导演状态】\n{director_context(state)}\n\n"
        f"【最近时间线】\n{state.get_recent_history(16)}\n\n"
        f"【用户干预】\n{raw_text}\n\n"
        f"【作用范围】\n{scope}"
    )
    response = llm.complete(f"{INTERVENTION_PROMPT}\n\n{user_prompt}", max_tokens=900)
    payload = _extract_json(response.text)
    payload.update({
        "raw_text": raw_text,
        "scope": scope,
        "created_at_turn": state.turn_count,
        "effective_after_turn": state.turn_count,
        "status": "preview",
        "expires_after_turns": expires_after_turns if scope == "turns" else (None if scope == "persistent" else 1),
    })
    intervention = Intervention.from_mapping(payload)
    return validate_intervention(state, intervention)


def has_blocking_conflicts(intervention: Intervention) -> bool:
    return any(item.get("severity") == "blocking" for item in intervention.conflicts)


def active_guidance(state: SimulationState) -> list[Intervention]:
    result = []
    for item in state.interventions:
        if item.mode != "guidance" or item.status not in {"pending", "applied"}:
            continue
        if item.status == "pending":
            result.append(item)
            continue
        if item.scope == "persistent":
            result.append(item)
        elif item.scope == "turns" and item.applied_at_turn is not None:
            if state.turn_count - item.applied_at_turn < (item.expires_after_turns or 1):
                result.append(item)
    return result


def pending_direct_event(state: SimulationState) -> Intervention | None:
    return next(
        (item for item in state.interventions if item.status == "pending" and item.mode in {"event", "override"}),
        None,
    )


def guidance_context(state: SimulationState) -> str:
    items = active_guidance(state)
    if not items:
        return "- 无"
    return "\n".join(f"- [{item.id}] {item.normalized_directive or item.raw_text}" for item in items)


def mark_guidance_applied(state: SimulationState) -> list[str]:
    """Return guidance IDs to commit alongside the next narration message."""

    return [item.id for item in active_guidance(state) if item.status == "pending"]


def intervention_message(state: SimulationState, item: Intervention) -> Message:
    return Message(
        speaker="导演",
        action="用户干预生效",
        speech=item.event_narration,
        turn=state.turn_count + 1,
        kind="intervention",
        visibility=item.visibility,
        visibility_scopes=[item.visibility],
        authoritative=True,
        state_patch=item.proposed_patch if item.visibility == "public" else [],
        intent={
            "action_type": "director_intervention",
            "intervention_id": item.id,
            "mode": item.mode,
        },
    )


def public_intervention(item: Intervention) -> dict[str, Any]:
    payload = asdict(item)
    payload["has_blocking_conflicts"] = has_blocking_conflicts(item)
    return payload
