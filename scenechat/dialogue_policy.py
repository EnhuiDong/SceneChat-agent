from __future__ import annotations

from typing import Any

from .models import AgentState
from .runtime import Intent


def _ratio(profile: dict[str, Any], key: str, default: float = 0.5) -> float:
    try:
        return max(0.0, min(float(profile.get(key, default)), 1.0))
    except (TypeError, ValueError):
        return default


def safe_obligation_fallback(
    agent: AgentState,
    pending: dict[str, Any],
    rejection: str,
) -> Intent:
    """Return a deterministic, non-factual reply after model retries are exhausted.

    The templates express only acknowledgement, boundaries or a request for
    clarification. They never invent an outcome, accept an action on another
    character's behalf, or reveal private profile content.
    """

    profile = agent.voice_profile if isinstance(agent.voice_profile, dict) else {}
    directness = _ratio(profile, "directness")
    politeness = _ratio(profile, "politeness")
    expressiveness = _ratio(profile, "emotional_expressiveness")
    requester = str(pending.get("speaker") or "对方").strip()
    move = str(pending.get("move") or "question")

    if move == "request":
        if directness >= 0.67:
            speech = "我听到了，但现在不能照办。"
        elif politeness >= 0.67:
            speech = "我明白你的要求，不过目前还不能答应。"
        else:
            speech = "这件事我听见了，但现在没法给出承诺。"
    elif move == "challenge":
        if directness >= 0.67:
            speech = "我听见了。先把依据说清楚。"
        elif politeness >= 0.67:
            speech = "我明白你的质疑，但现在还不能给出完整说明。"
        else:
            speech = "你的质疑我听见了；这件事暂时不能完整回答。"
    elif directness >= 0.67:
        speech = "我听见了，但现在不能完整回答。"
    elif politeness >= 0.67:
        speech = "我明白你的问题，不过目前只能先回应到这里。"
    else:
        speech = "这件事我听见了，但现在还不能给出完整答复。"

    action = (
        f"明显停顿后将注意力转向{requester}。"
        if expressiveness >= 0.67
        else f"将注意力转向{requester}，作出有限回应。"
    )
    try:
        urgency = float(pending.get("urgency", 0.0) or 0.0)
    except (TypeError, ValueError):
        urgency = 0.0
    return Intent(
        actor=agent.name,
        action_type="speak",
        action=action,
        speech=speech,
        private_reason=f"模型重试后仍未完成必要回应：{rejection}"[:500],
        addressed_to=[requester] if requester else [],
        reply_to_event_id=str(pending.get("event_id") or ""),
        thread_id=str(pending.get("thread_id") or ""),
        reply_to_obligation_id=str(pending.get("obligation_id") or ""),
        obligation_resolution="responded",
        conversation_move="deflect",
        urgency=urgency,
    )
