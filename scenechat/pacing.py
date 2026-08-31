from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Message, SimulationState


@dataclass(frozen=True)
class PacingPolicy:
    pace: int
    label: str
    narration_interval: int
    stagnation_limit: int
    active_beat_limit: int
    horizon_multiplier: float
    direction: str

    @classmethod
    def from_value(cls, value: int) -> "PacingPolicy":
        pace = max(0, min(int(value), 100))
        if pace <= 20:
            return cls(pace, "沉浸", 5, 7, 1, 1.65, "放慢推进，充分呈现反应、关系和局部细节")
        if pace <= 40:
            return cls(pace, "舒缓", 4, 6, 1, 1.3, "保留余韵，以人物反应为主，适度推进")
        if pace <= 60:
            return cls(pace, "均衡", 3, 5, 1, 1.0, "平衡人物互动、事件变化和目标推进")
        if pace <= 80:
            return cls(pace, "紧凑", 2, 3, 2, 0.75, "减少重复试探，让行动产生清晰后果并推动节点")
        return cls(pace, "冲刺", 1, 2, 3, 0.55, "合并过渡，优先触发关键转折并自然接近结局")


def _beats(state: SimulationState) -> list[Any]:
    return list(getattr(state.world_spec, "beat_specs", []) or [])


def initialize_arc(state: SimulationState, *, reset_horizon: bool = False) -> None:
    """Initialize the target horizon and currently eligible beats."""

    policy = PacingPolicy.from_value(state.arc_state.pace)
    beats = _beats(state)
    if reset_horizon or state.arc_state.target_end_turn is None:
        base = max(12, len(state.agents) * 4 + max(1, len(beats)) * 6)
        remaining = max(6, round(base * policy.horizon_multiplier))
        state.arc_state.target_end_turn = state.turn_count + remaining
    refresh_active_beats(state)
    if not beats and not state.ended:
        horizon = max(state.turn_count + 1, state.arc_state.target_end_turn or 1)
        state.arc_state.progress = min(0.92, state.turn_count / horizon)


def refresh_active_beats(state: SimulationState) -> None:
    beats = _beats(state)
    resolved = set(state.arc_state.resolved_beat_ids)
    skipped = set(state.arc_state.skipped_beat_ids)
    eligible = [
        beat.id
        for beat in beats
        if beat.id not in resolved
        and beat.id not in skipped
        and set(beat.prerequisites).issubset(resolved)
        and (not beat.phase_hint or beat.phase_hint == state.current_phase)
    ]
    limit = PacingPolicy.from_value(state.arc_state.pace).active_beat_limit
    state.arc_state.active_beat_ids = eligible[:limit]


def active_beat_context(state: SimulationState) -> str:
    initialize_arc(state)
    by_id = {beat.id: beat for beat in _beats(state)}
    lines = []
    for beat_id in state.arc_state.active_beat_ids:
        beat = by_id.get(beat_id)
        if beat is None:
            continue
        marker = "必须保留" if beat.required else "目标节点"
        signals = f"；可判定信号：{'、'.join(beat.resolution_signals)}" if beat.resolution_signals else ""
        lines.append(f"- {beat.id}（{marker}）：{beat.description}{signals}")
    return "\n".join(lines) or "- 暂无明确节点；依据当前冲突自然推进"


def pacing_context(state: SimulationState) -> str:
    initialize_arc(state)
    policy = PacingPolicy.from_value(state.arc_state.pace)
    return (
        f"节奏档位：{policy.label}（{policy.pace}/100）。{policy.direction}。\n"
        f"剧情进度：{round(state.arc_state.progress * 100)}%；张力：{round(state.arc_state.tension * 100)}%。\n"
        f"预计收束轮次：约第 {state.arc_state.target_end_turn} 轮。该数字是软目标，不得牺牲人物逻辑或硬规则。\n"
        f"当前可推进节点：\n{active_beat_context(state)}"
    )


def should_insert_narration(state: SimulationState) -> bool:
    initialize_arc(state)
    if not state.history or state.history[-1].kind in {"narration", "intervention"}:
        return False
    policy = PacingPolicy.from_value(state.arc_state.pace)
    if state.arc_state.turns_since_progress >= policy.stagnation_limit:
        return True
    return state.agent_turn_count > 0 and state.agent_turn_count % policy.narration_interval == 0


def required_beats_resolved(state: SimulationState, additional: list[str] | None = None) -> bool:
    resolved = set(state.arc_state.resolved_beat_ids) | set(additional or [])
    return all(not beat.required or beat.id in resolved for beat in _beats(state))


def validate_resolved_beats(state: SimulationState, values: Any) -> list[str]:
    requested = [str(item) for item in values or []]
    active = set(state.arc_state.active_beat_ids)
    return [beat_id for beat_id in requested if beat_id in active]


def update_arc_after_message(state: SimulationState, message: Message) -> None:
    initialize_arc(state)
    arc_updates = message.intent.get("arc_updates", {}) if isinstance(message.intent, dict) else {}
    resolved_now = validate_resolved_beats(state, arc_updates.get("resolved_beat_ids", []))
    previous = set(state.arc_state.resolved_beat_ids)
    for beat_id in resolved_now:
        if beat_id not in previous:
            state.arc_state.resolved_beat_ids.append(beat_id)
            previous.add(beat_id)

    if resolved_now:
        state.arc_state.turns_since_progress = 0
    else:
        state.arc_state.turns_since_progress += 1

    try:
        proposed_tension = float(arc_updates.get("tension", state.arc_state.tension))
        state.arc_state.tension = max(0.0, min(proposed_tension, 1.0))
    except (TypeError, ValueError):
        pass

    beats = _beats(state)
    if beats:
        total_weight = sum(max(1, beat.weight) for beat in beats)
        resolved_weight = sum(
            max(1, beat.weight) for beat in beats if beat.id in previous
        )
        state.arc_state.progress = min(1.0, resolved_weight / total_weight)
    elif state.ended:
        state.arc_state.progress = 1.0
    else:
        horizon = max(state.turn_count + 1, state.arc_state.target_end_turn or 1)
        state.arc_state.progress = min(0.92, state.turn_count / horizon)
    if state.ended:
        state.arc_state.progress = 1.0
    refresh_active_beats(state)
