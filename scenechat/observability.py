"""Director-facing, bounded views over committed simulation state.

The payload deliberately exposes structured state and evidence rather than model
scratch work.  It is intended for the local story owner/director UI, not for an
individual character's context.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .evaluation import evaluate_trace
from .models import Message, SimulationState


QUALITY_SIGNALS = {
    "conversation_responsiveness": (
        "回应闭环",
        "直接问题、请求和挑战是否得到明确回应。",
    ),
    "threaded_interaction": (
        "议题连贯",
        "角色行动是否能延续已有议题，而非频繁另起话题。",
    ),
    "relationship_evidence": (
        "关系有据",
        "关系变化是否引用了角色实际可见的事件。",
    ),
    "relationship_evidence_reuse": (
        "证据不过用",
        "同一事件是否避免被重复用于推动同一关系维度。",
    ),
    "belief_provenance": (
        "认知可追溯",
        "人物认知是否保留来源或明确标记为自身推断。",
    ),
    "quality_gate_stability": (
        "对话稳定",
        "生成结果是否较少触发重复、漏回应或泄密重试。",
    ),
    "generation_stability": (
        "生成稳定",
        "本次推演是否较少出现解析失败、降级行动或阻塞。",
    ),
    "narration_freshness": (
        "旁白新鲜度",
        "近期旁白是否避免重复同一广播、环境描写或加压动作。",
    ),
    "agent_event_share": (
        "角色参与度",
        "时间线是否由角色行动推动，而不是被旁白长期占据。",
    ),
    "scheduler_liveness": (
        "调度可继续",
        "仍有可行动人物时，当前阶段是否始终保留下一位行动者。",
    ),
    "arc_momentum": (
        "剧情节点动量",
        "目标节点是否在合理轮数内得到实质推进或结算。",
    ),
}


def _event_preview(message: Message | None) -> dict[str, Any] | None:
    if message is None:
        return None
    return {
        "event_id": message.event_id,
        "turn": message.turn,
        "speaker": message.speaker,
        "kind": message.kind,
        "action": message.action[:180],
        "speech": message.speech[:240],
    }


def _quality_payload(state: SimulationState) -> list[dict[str, Any]]:
    if not state.history:
        return []
    metrics = evaluate_trace(
        state.history,
        expected_characters=state.agent_order,
        state=state,
        expect_end=state.ended,
    )
    result = []
    for key, (label, description) in QUALITY_SIGNALS.items():
        score = metrics.get(key)
        if score is None:
            continue
        result.append({
            "id": key,
            "label": label,
            "description": description,
            "value": score.value,
            "passed": score.passed,
        })
    return result


def director_observability_payload(
    state: SimulationState,
    *,
    include_quality: bool = True,
) -> dict[str, Any]:
    """Return a compact owner-only explanation of committed runtime state."""

    event_index = {message.event_id: message for message in state.history[-80:]}
    scheduler = dict(state.last_scheduler_decision)
    source_event_id = str(scheduler.get("source_event_id") or "")
    scheduler["source_event"] = _event_preview(event_index.get(source_event_id))

    thread_order = {"active": 0, "dormant": 1, "resolved": 2}
    threads = []
    for thread in sorted(
        state.conversation_threads.values(),
        key=lambda item: (
            thread_order.get(item.status, 3),
            -item.last_active_turn,
        ),
    )[:12]:
        obligations = [
            {
                "id": item.id,
                "requester": item.requester,
                "target": item.target,
                "move": item.move,
                "summary": item.summary,
                "urgency": item.urgency,
                "status": item.status,
                "source_event_id": item.source_event_id,
                "resolution_event_id": item.resolution_event_id,
                "created_at_turn": item.created_at_turn,
                "updated_at_turn": item.updated_at_turn,
            }
            for item in thread.obligations[-12:]
        ]
        pressures = [
            {
                "agent": agent.name,
                "value": agent.disclosure_pressure_by_thread.get(thread.id, 0.0),
            }
            for agent in state.agents.values()
            if thread.id in agent.disclosure_pressure_by_thread
        ]
        threads.append({
            "id": thread.id,
            "topic": thread.topic,
            "status": thread.status,
            "participants": list(thread.participants),
            "tension": thread.tension,
            "created_at_turn": thread.created_at_turn,
            "last_active_turn": thread.last_active_turn,
            "obligations": obligations,
            "pressures": pressures,
        })

    relationships = []
    for observer in state.agents.values():
        for target, dynamic in observer.relationship_dynamics.items():
            if not isinstance(dynamic, dict):
                continue
            facets = []
            for dimension_id, raw_value in (dynamic.get("facets") or {}).items():
                dimension = state.relationship_dimensions.get(dimension_id)
                if dimension is None:
                    continue
                try:
                    value = max(0.0, min(float(raw_value), 1.0))
                except (TypeError, ValueError):
                    continue
                facets.append({
                    "id": dimension_id,
                    "label": dimension.get("label", dimension_id),
                    "low_label": dimension.get("low_label", "低"),
                    "high_label": dimension.get("high_label", "高"),
                    "value": round(value, 4),
                })
            evidence_items = [
                item for item in (dynamic.get("evidence") or [])
                if isinstance(item, dict)
            ]
            latest = evidence_items[-1] if evidence_items else None
            latest_payload = None
            if latest is not None:
                changes = []
                proposed = latest.get("proposed_facets") or {}
                for dimension_id, raw_delta in (latest.get("facets") or {}).items():
                    dimension = state.relationship_dimensions.get(dimension_id)
                    if dimension is None:
                        continue
                    changes.append({
                        "id": dimension_id,
                        "label": dimension.get("label", dimension_id),
                        "delta": raw_delta,
                        "proposed": proposed.get(dimension_id, raw_delta),
                    })
                evidence_event_id = str(latest.get("event_id") or "")
                latest_payload = {
                    "event_id": evidence_event_id,
                    "turn": latest.get("turn", 0),
                    "note": str(latest.get("note") or "")[:300],
                    "cap": latest.get("cap", 0.0),
                    "changes": changes,
                    "source_event": _event_preview(event_index.get(evidence_event_id)),
                }
            if facets or latest_payload:
                relationships.append({
                    "observer": observer.name,
                    "target": target,
                    "summary": str(observer.relationships.get(target) or "")[:300],
                    "facets": facets,
                    "latest_evidence": latest_payload,
                })
    relationships.sort(
        key=lambda item: (
            -int((item.get("latest_evidence") or {}).get("turn") or 0),
            item["observer"],
            item["target"],
        )
    )

    agents = []
    for agent in state.agents.values():
        status_counts = Counter(
            item.epistemic_status for item in agent.belief_records if item.active
        )
        recent_beliefs = [
            {
                "id": item.id,
                "content": item.content,
                "source_agent": item.source_agent,
                "source_event_id": item.source_event_id,
                "epistemic_status": item.epistemic_status,
                "confidence": item.confidence,
                "created_at_turn": item.created_at_turn,
            }
            for item in agent.belief_records
            if item.active
        ][-5:]
        agents.append({
            "name": agent.name,
            "emotion": agent.current_emotion,
            "emotion_intensity": agent.emotion_intensity,
            "conversation_goal": agent.current_conversation_goal,
            "core_beliefs": list(agent.core_beliefs),
            "pending_obligation_count": len(agent.pending_intents),
            "active_thread_count": len(state.active_threads_for(agent.name)),
            "belief_status_counts": dict(status_counts),
            "recent_beliefs": recent_beliefs,
        })

    all_obligations = [
        obligation
        for thread in state.conversation_threads.values()
        for obligation in thread.obligations
    ]
    payload = {
        "turn": state.turn_count,
        "scheduler": scheduler,
        "thread_summary": {
            "active": sum(item.status == "active" for item in state.conversation_threads.values()),
            "dormant": sum(item.status == "dormant" for item in state.conversation_threads.values()),
            "resolved": sum(item.status == "resolved" for item in state.conversation_threads.values()),
            "open_obligations": sum(item.status == "open" for item in all_obligations),
            "responded_unresolved": sum(item.status == "responded" for item in all_obligations),
        },
        "relationship_dimensions": [
            {"id": key, **value}
            for key, value in state.relationship_dimensions.items()
        ],
        "threads": threads,
        "relationships": relationships[:16],
        "agents": agents,
    }
    if include_quality:
        payload["quality_signals"] = _quality_payload(state)
    return payload
