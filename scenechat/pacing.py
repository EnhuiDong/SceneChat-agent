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
            return cls(pace, "紧凑", 3, 3, 2, 0.75, "减少重复试探，优先执行已有方案并呈现真实后果")
        return cls(pace, "冲刺", 4, 2, 3, 0.55, "压缩过渡，优先完成尚未执行的选择；不增加旁白来替代行动")


def _beats(state: SimulationState) -> list[Any]:
    return list(getattr(state.world_spec, "beat_specs", []) or [])


def arc_view(state):
    from dataclasses import asdict
    refresh_active_beats(state)
    beats = _beats(state)
    events = {m.event_id: m for m in state.history}
    def evidence(beat):
        items = []
        for event_id in state.arc_state.beat_records.get(beat.id, {}).get("evidence_event_ids", []):
            event = events.get(event_id)
            if event is None:
                continue
            visible = event.visibility in {"public", "audience_only"}
            items.append({"id": event_id, "turn": event.turn, "visible": visible,
                          "speaker": event.speaker if visible else "",
                          "excerpt": (event.speech or event.action)[:90] if visible else ""})
        return items
    return {
        **asdict(state.arc_state),
        "progress_mode": "milestones" if beats else "open_ended",
        "completed_count": sum(b.id in state.arc_state.resolved_beat_ids for b in beats),
        "total_count": len(beats),
        "beats": [{"id": b.id, "description": b.description,
                   **state.arc_state.beat_statuses.get(b.id, {}),
                   "completion": state.arc_state.beat_records.get(b.id), "last_check": state.arc_state.beat_checks.get(b.id), "evidence": evidence(b)} for b in beats],
    }


def initialize_arc(state: SimulationState, *, reset_horizon: bool = False) -> None:
    """Initialize the target horizon and currently eligible beats."""

    policy = PacingPolicy.from_value(state.arc_state.pace)
    beats = _beats(state)
    if reset_horizon or state.arc_state.target_end_turn is None:
        base = max(12, len(state.agents) * 4 + max(1, len(beats)) * 6)
        remaining = max(6, round(base * policy.horizon_multiplier))
        state.arc_state.target_end_turn = state.turn_count + remaining
    refresh_active_beats(state)
    if not beats:
        state.arc_state.progress = 0.0


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
    for beat in beats:
        blocked = ""
        if not set(beat.prerequisites).issubset(resolved): blocked = "前置节点尚未完成"
        elif beat.phase_hint and beat.phase_hint != state.current_phase: blocked = "等待适用阶段"
        status = "completed" if beat.id in resolved else "skipped" if beat.id in skipped else "in_progress" if beat.id in state.arc_state.active_beat_ids else "available"
        if status in {"completed", "skipped"}:
            blocked = ""
        state.arc_state.beat_statuses[beat.id] = {"status": status, "blocked_reason": blocked}


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
        lines.append(f"- {beat.id}（{marker}，{beat.completion_mode}）：{beat.description}{signals}；条件={beat.completion_conditions}")
    return "\n".join(lines) or "- 暂无明确节点；依据当前冲突自然推进"


def pacing_context(state: SimulationState) -> str:
    initialize_arc(state)
    policy = PacingPolicy.from_value(state.arc_state.pace)
    resolution_instruction = ""
    if policy.pace > 80:
        resolution_instruction = (
            "\n本档位要求：角色优先执行已有可行方案、结束无新证据的争论；旁白只呈现已完成的结果。"
            "缺少决定、资源或证据时仍须真实取得，不能由旁白宣布完成。"
        )
    elif policy.pace > 60:
        resolution_instruction = (
            "\n本档位要求：优先制造能实质推进当前节点的机会，并在结果已经发生时准确提交 resolved_beat_ids。"
        )
    return (
        f"节奏档位：{policy.label}（{policy.pace}/100）。{policy.direction}。\n"
        f"已完成节点：{len(state.arc_state.resolved_beat_ids)}/{len(_beats(state))}（无节点表示开放推进）；张力：{round(state.arc_state.tension * 100)}%。\n"
        f"预计收束轮次：约第 {state.arc_state.target_end_turn} 轮。该数字是软目标，不得牺牲人物逻辑或硬规则。\n"
        f"当前可推进节点：\n{active_beat_context(state)}"
        f"{resolution_instruction}\n任何节奏均不得跳过必要选择、信息获取和规则结算；慢速仍须产生新的反应或关系变化，不得重复问答。"
    )


def should_insert_narration(state: SimulationState) -> bool:
    initialize_arc(state)
    if not state.history or state.history[-1].kind in {"narration", "intervention"}:
        return False
    phase = state.phase_specs.get(state.current_phase)
    if phase is not None and phase.advance_when == "all_active_voted":
        return False  # Ballots are actor events; only the resolver may announce the tally.
    actors = [m for m in state.history if m.speaker in state.agents][-3:]
    if len(actors) == 3 and all(m.intent.get("action_type") == "pass" and not m.speech and not m.state_patch for m in actors):
        return False  # Waiting silently is not material for another waiting scene.
    if actors and all(
        message.intent.get("action_type") in {"speak", "pass", "observe"}
        and not message.intent.get("meaningful_state_change")
        for message in actors
    ):
        return False  # Dialogue-only rounds do not need recurring empty atmosphere.
    policy = PacingPolicy.from_value(state.arc_state.pace)
    # A stalled discussion needs a different actor decision, not another timer.
    return state.agent_turn_count > 0 and state.agent_turn_count % policy.narration_interval == 0


def required_beats_resolved(state: SimulationState, additional: list[str] | None = None) -> bool:
    resolved = set(state.arc_state.resolved_beat_ids) | set(additional or [])
    return all(not beat.required or beat.id in resolved for beat in _beats(state))


def validate_resolved_beats(state: SimulationState, values: Any, verified=None) -> list[str]:
    requested = [str(item) for item in values or []]
    active = set(state.arc_state.active_beat_ids)
    if getattr(state.world_spec, "execution_version", 1) == 1:
        return list(dict.fromkeys(beat_id for beat_id in requested if beat_id in active))
    return list(dict.fromkeys(beat_id for beat_id in requested if beat_id in active and beat_id in (verified or {})))


def update_arc_after_message(state: SimulationState, message: Message, *, previous_phase=None) -> None:
    initialize_arc(state)
    arc_updates = (message.intent.get("arc_updates", {})
                   if isinstance(message.intent, dict) and not message.intent.get("generation_fallback") else {})
    verified = arc_updates.get("verified_beats", {})
    resolved_now = validate_resolved_beats(state, arc_updates.get("resolved_beat_ids", []), verified)
    if getattr(state.world_spec, "execution_version", 1) == 2:
        from .beat_evidence import state_evidence
        for beat in _beats(state):
            eligible = (
                beat.id not in state.arc_state.resolved_beat_ids
                and beat.id not in state.arc_state.skipped_beat_ids
                and set(beat.prerequisites).issubset(state.arc_state.resolved_beat_ids)
                and (not beat.phase_hint or beat.phase_hint in {state.current_phase, previous_phase})
            )
            if eligible and beat.completion_mode == "state":
                evidence = state_evidence(state, beat)
                if evidence:
                    resolved_now.append(beat.id)
                    verified[beat.id] = {"method": "state", "evidence_event_ids": evidence, "reason": "声明条件已由提交的行动满足"}
    previous = set(state.arc_state.resolved_beat_ids)
    for beat_id in resolved_now:
        if beat_id not in previous:
            state.arc_state.resolved_beat_ids.append(beat_id)
            state.arc_state.beat_records.setdefault(beat_id, {
                **verified.get(beat_id, {"method": "legacy", "evidence_event_ids": []}),
                "completed_at_turn": message.turn,
            })
            previous.add(beat_id)

    meaningful = bool(message.authoritative and message.kind != "narration" and (
        message.intent.get("meaningful_state_change", False)
        or (not message.intent.get("generation_fallback")
            and message.intent.get("obligation_resolution") in {"satisfied", "withdrawn"})
    ))
    if message.kind == "intervention" or any(item.applied_at_turn == message.turn for item in state.interventions):
        state.arc_state.plan_adjusted_at_turn = message.turn
    if resolved_now or meaningful:
        state.arc_state.turns_since_progress = 0
    else:
        state.arc_state.turns_since_progress += 1

    try:
        proposed_tension = float(arc_updates.get("tension", state.arc_state.tension))
        # Older and compatible models sometimes return a percentage despite
        # the ratio schema.  Accept 25/100 as 0.25 while keeping 0..1 intact.
        if 1.0 < proposed_tension <= 100.0:
            proposed_tension /= 100.0
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
    else:
        state.arc_state.progress = 0.0
    refresh_active_beats(state)
