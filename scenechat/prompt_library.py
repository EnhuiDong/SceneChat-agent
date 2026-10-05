"""Human-readable prompt contracts, combined with typed schemas at call sites."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=3)
def prompt_contract(name):
    if name not in {"actor", "narrator", "scenario"}:
        raise ValueError("Unknown prompt contract")
    return (Path(__file__).parent / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def actor_worklist(state, agent):
    """Local/visible history only; no director goals or other private profiles."""
    phase = state.phase_specs.get(state.current_phase)
    from .pacing import PacingPolicy
    pace = PacingPolicy.from_value(state.arc_state.pace)
    own = [m for m in state.history if m.speaker == agent.name][-4:]
    pending = sorted(agent.pending_intents, key=lambda p: p.get("created_at_turn", 0), reverse=True)[:3]
    visible_actors = set()
    for message in reversed(state.history):
        if any(op.get("op") == "set_phase" or op.get("op") == "set_world" and op.get("key") == "current_phase"
               for op in message.state_patch):
            break
        if state._agent_can_observe(agent, message):
            visible_actors.add(message.speaker)
    lines = [f"用户节奏：{pace.label}（{pace.pace}/100），{pace.direction}。这不是新增世界规则。",
             f"真实阶段：{state.current_phase}；合法动作：{getattr(phase, 'allowed_action_types', []) or '自由行动'}",
             f"本阶段已行动（仅可观察者）：{'、'.join(sorted(state.phase_action_log & visible_actors)) or '无可见记录'}",
             "自己已经表达或尝试过（不要只换词重演）："]
    lines.extend(f"- {m.event_id}：{m.action[:100]} {m.speech[:240]}" for m in own)
    lines.append("尚待回应的事项（回应不等于解决）：")
    lines.extend(f"- {p.get('event_id')}：{p.get('speaker')} / {str(p.get('summary', ''))[:200]}" for p in pending)
    from .agenda import relevant_tasks
    tasks = relevant_tasks(state, agent)
    lines.append("当前相关事项（只记录已提交事件；回应不是事实核验，也不要求所有人同意）：")
    lines.extend(
        f"- {task.id} [{task.status}] {task.title[:140]}；下一步：{task.next_action}；证据事件：{','.join(task.evidence_event_ids[-3:])}"
        for task in tasks
    )
    if not tasks:
        lines.append("- 无关联的开放事项；按自身目标与当前阶段选择具体行动。")
    decision = state.last_scheduler_decision
    if decision.get("actor_name") == agent.name and decision.get("source_event_id"):
        lines.append(f"本次调度希望你处理的事件：{decision['source_event_id']}；优先实际回应其内容，不再征求如何开始回应。")
    recent = [m for m in state.history if m.kind != "narration" and state._agent_can_observe(agent, m)][-4:]
    lines.append("最新可见行动/回答（优先于旧待办的措辞；台词仍是主张）：")
    lines.extend(f"- {m.event_id} {m.speaker}：{m.action[:70]} {m.speech[:220]}" for m in recent)
    settled_moves = {"answer", "support", "acknowledge"}
    if (not pending and len(recent) >= 2 and all(
        m.intent.get("conversation_move") in settled_moves
        and m.intent.get("action_type", "speak") == "speak"
        and not m.state_patch
        for m in recent[-2:]
    )):
        lines.append(
            "最近两次可见回合都在答应或附和，当前话题可能已谈妥。"
            "除非有新的实际分歧、证据或代价，不再重复确认；能行动就做第一步，"
            "不能行动可转向另一件具体未办事项或自然结束这一交流。"
        )
    if phase and phase.next_phase:
        lines.append(f"阶段出口：{phase.advance_when} → {phase.next_phase}；必须经真实行动触发，不能由发言宣告代替。")
        if phase.advance_when == "manual" and pace.pace > 60:
            lines.append("若当前讨论已无可取得的新依据，评估合法的阶段出口规则；不必等所有人再次表态同意。仍须遵守已声明的前置条件。")
    return "\n".join(lines)


def narrator_facts(state):
    import json
    from .visibility import can_access, ViewerContext
    public = ViewerContext()
    events = [m for m in state.history[-16:] if m.authoritative and m.kind != "narration"
              and can_access(m.scopes, public)]
    # These operations are public only when the source event is public. Never
    # expose arbitrary world/resource fields or private knowledge patches.
    public_ops = {"record_vote", "clear_votes", "set_phase", "move_agent", "set_agent_status"}
    from .narrative_grounding import _latest_vote_batch
    committed = {m.event_id: m for m in state.history if m.authoritative and m.kind != "narration"}
    publicly_inactive = set()
    for message in committed.values():
        if not can_access(message.scopes, public):
            continue
        for operation in message.state_patch:
            if operation.get("op") != "set_agent_status" or operation.get("key") not in {"alive", "active"}:
                continue
            target = operation.get("target")
            if target not in state.agents:
                continue
            if operation.get("value") is False:
                publicly_inactive.add(target)
            elif operation.get("value") is True:
                publicly_inactive.discard(target)
    actual_inactive = {agent.name for agent in state.agents.values() if not agent.eligible}
    active_names = ([agent.name for agent in state.agents.values() if agent.eligible]
                    if publicly_inactive == actual_inactive else None)
    ballots, closing, sources = _latest_vote_batch(committed.values(), state.agents)
    vote_readback = None
    if (ballots and all(can_access(committed[source].scopes, public) for source in sources)
            and (not closing or can_access(committed[closing].scopes, public))):
        vote_readback = {"ballots": ballots, "counts": {name: list(ballots.values()).count(name) for name in set(ballots.values())},
                         "completed": bool(closing), "closing_event_id": closing,
                         "closing_turn": committed[closing].turn if closing else None,
                         "instruction": "此处由引擎计数。未完成时不宣布整轮结果；出局不等于身份公开。"}
    return json.dumps({
        "phase": state.current_phase,
        "public_vote_readback": vote_readback,
        "cast": [a.name for a in state.agents.values()],
        "public_active_cast": active_names,
        "public_active_count": len(active_names) if active_names is not None else None,
        "director_mutable_fields": [key for key, field in state.state_schema.items()
                                    if "director" in field.mutable_by and "public" in field.visibility
                                    and key not in {"current_phase", "round_number"}],
        "committed_events": [{"event_id": m.event_id, "action": m.action,
                              "committed_public_effects": [op for op in m.state_patch if op.get("op") in public_ops],
                              "public_changes": m.public_changes} for m in events],
    }, ensure_ascii=False)
