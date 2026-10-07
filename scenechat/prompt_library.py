"""Human-readable prompt contracts, combined with typed schemas at call sites."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def prompt_contract(name):
    if name not in {"actor", "dialogue", "performance", "characters", "narrator", "scenario", "scene_actor", "execution_design"}:
        raise ValueError("Unknown prompt contract")
    return (Path(__file__).parent / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def actor_exchange_events(state, agent):
    """Single source of truth for the exchange window, including old replies."""
    visible = [m for m in state.history if state._agent_can_observe(agent, m)]
    pending_ids = {str(p.get("event_id") or "") for p in agent.pending_intents}
    decision = state.last_scheduler_decision
    scheduled_id = (str(decision.get("source_event_id") or "")
                    if decision.get("actor_name") == agent.name else "")
    sources = [m for m in visible if m.event_id in pending_ids or m.event_id == scheduled_id]
    focus = next((m for m in reversed(sources) if m.event_id == scheduled_id), None)
    focus = focus or (sources[-1] if sources else None)
    recent = visible[-6:]
    if focus is not None and focus not in recent:
        recent = [focus, *recent]
    return recent, focus


def actor_exchange(state, agent):
    """Present visible dialogue last; include an older pending source separately."""
    recent, focus = actor_exchange_events(state, agent)
    from .conversation_recall import render_recall
    receipts = render_recall(state, agent, recent, focus)
    lines = [receipts] if receipts else []
    lines.append("【眼前的近期交流】这是实际发生过的可见交流。先听这几个人正在说什么，再决定自己的反应；台词不是客观事实。")
    for message in recent:
        lines.append(f"- [{message.event_id}] {message.as_observation_for(agent.name)[:1240]}")
        effects = visible_action_readback(state, agent, message)
        if effects:
            lines.append("  该事件确已执行（不是台词中的打算）：" + "；".join(effects))
    if not recent:
        lines.append("- 尚无可见的对话。按开场处境反应，不把未发生的讨论或关系当作已有历史。")
    if focus is not None and focus.speaker == agent.name:
        lines.append(f"本轮回到你先前提出的事 [{focus.event_id}]；结合已经收到的回应，决定接下来做什么、改条件或放下，不是重新把同一个请求再问一遍。")
    elif focus is not None:
        lines.append(f"本轮应回应 {focus.speaker} 这次说或做的具体内容 [{focus.event_id}]；可以答、拒绝或有限回应，无须把整场争论总结一遍，也不用宣布自己将要作答。")
    lines.append("时间向前：开场写在桌上的东西、准备中的动作或初始站位，不会因再次读到背景就恢复原样；以你能看见的后续事件和已提交状态为准。新打算不算已完成，别人说过的话也不自动成为世界事实。")
    own = [m for m in state.history if m.speaker == agent.name and m.kind == "dialogue"
           and not m.intent.get("generation_fallback")]
    if not own:
        lines.append("这是你的第一次开口。是否认识别人以设定为准；其他人已讨论过，不等于你也已经发表过立场。")
    else:
        lines.append(f"你上次实际说的是：{own[-1].speech[:500] or '没有说话'}。承接其态度；新事件可以让你改口，但不要每回合重新开始人物。")
        previous_concern = str(own[-1].intent.get("private_reason") or "").strip()
        if previous_concern and not own[-1].intent.get("generation_fallback"):
            lines.append(f"上次自己的未说出口的顾虑（仅你可见，不是世界事实）：{previous_concern[:240]}。这份顾虑是否仍在，由后续事件判断，不照搬成台词。")
    phase = state.phase_specs.get(state.current_phase)
    lines.append(f"现在实际处于：{state.current_phase}。前一阶段的话题不会让当前阶段倒退。")
    if phase and phase.allowed_action_types == ["vote"]:
        lines.append("本阶段需提交自己的投票选择；可以继续回应别人，但 action 必须交代本次实际选择，不要一边仅作介绍或看向某人，一边在执行字段悄悄投给另一个人。沉默投票可以不说理由，但须写出自己的投票动作。")
    return "\n".join(lines)


def visible_action_readback(state, agent, message):
    """Expose committed, observable choices, not arbitrary private patches.

    A model may speak about X while voting for Y in its execution fields.
    Subsequent actors must see that actual public choice, not only the prose.
    Individual perception overrides and private scopes remain authoritative.
    """
    from .visibility import ViewerContext, can_access
    if (not message.authoritative or not state._agent_can_observe(agent, message)
            or agent.name in message.individual_observations):
        return []
    viewer = ViewerContext(name=agent.name, role=agent.role, location=agent.current_location)
    if not can_access(message.scopes, viewer):
        return []
    effects = []
    for op in message.state_patch:
        if op.get("op") == "record_vote":
            effects.append(f"{op.get('actor')} 投票给 {op.get('target')}")
        elif op.get("op") == "set_phase":
            if op.get("value") in state.phase_specs:
                effects.append(f"阶段切换到 {op.get('value')}")
        elif op.get("op") == "clear_votes":
            effects.append("该轮选票已清空")
        elif op.get("op") == "move_agent":
            # This is the movement in an observable submitted event, not a
            # global lookup of private current positions. Overrides above
            # prevent an altered perception from acquiring the real result.
            if op.get("target") in state.agents and op.get("value") in state.locations:
                effects.append(f"{op['target']} 已移动到 {op['value']}")
        elif op.get("op") == "set_entity":
            spec = getattr(state.world_spec, "entities", {}).get(op.get("target"))
            if (isinstance(spec, dict) and can_access(spec.get("visibility"), viewer)
                    and op.get("key") in (spec.get("state") or {})):
                import json
                effects.append(f"{op['target']}.{op['key']} 已设置为 "
                               + json.dumps(op.get("value"), ensure_ascii=False))
        elif op.get("op") in {"set_world", "increment_world"}:
            field = state.state_schema.get(op.get("key"))
            if field and can_access(field.visibility, viewer):
                import json
                if op['op'] == "set_world":
                    effects.append(f"{op['key']} 已设置为 " + json.dumps(op.get("value"), ensure_ascii=False))
                else:
                    effects.append(f"{op['key']} 在此事件中变化了 " + json.dumps(op.get("amount"), ensure_ascii=False))
    return effects


def actor_worklist(state, agent, *, presented_events=()):
    """Local/visible history only; no director goals or other private profiles."""
    phase = state.phase_specs.get(state.current_phase)
    from .pacing import PacingPolicy
    pace = PacingPolicy.from_value(state.arc_state.pace)
    presented_ids = {m.event_id for m in presented_events}
    own = [m for m in state.history if m.speaker == agent.name
           and not m.intent.get("generation_fallback")
           and state._agent_can_observe(agent, m)][-4:]
    own = [m for m in own if m.event_id not in presented_ids]
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
    lines.extend(f"- {m.event_id}：{m.as_observation_for(agent.name)[:340]}" for m in own)
    if not own:
        lines.append("- 最近的本人言行已在末尾交流中列出，不重复抄写。")
    lines.append("尚待回应的事项（回应不等于解决）：")
    lines.extend(f"- {p.get('event_id')}：{p.get('speaker')} / {str(p.get('summary', ''))[:200]}" for p in pending)
    from .agenda import relevant_tasks
    tasks = relevant_tasks(state, agent)
    lines.append("当前相关事项（只记录已提交事件；回应不是事实核验，也不要求所有人同意）：")
    for task in tasks:
        title = "原话见末尾交流" if task.source_event_id in presented_ids else task.title[:140]
        evidence = [event_id for event_id in task.evidence_event_ids[-3:]
                    if any(m.event_id == event_id and state._agent_can_observe(agent, m) for m in state.history)]
        lines.append(f"- {task.id} [{task.status}] {title}；来源事件：{task.source_event_id}；证据事件：{','.join(evidence)}")
    lines.append("事项状态只记录回应是否收到，不规定你必须核验、主持、合作或重新提问；下一步由你的目标和性格决定。")
    if not tasks:
        lines.append("- 无关联的开放事项；按自身目标与当前阶段选择具体行动。")
    decision = state.last_scheduler_decision
    if decision.get("actor_name") == agent.name and decision.get("source_event_id"):
        lines.append(f"本次调度希望你处理的事件：{decision['source_event_id']}；优先实际回应其内容，不再征求如何开始回应。")
    recent = [m for m in state.history if m.kind != "narration" and state._agent_can_observe(agent, m)][-4:]
    unpresented = [m for m in recent if m.event_id not in presented_ids]
    if unpresented:
        lines.append("最新可见行动/回答（优先于旧待办的措辞；台词仍是主张）：")
        lines.extend(f"- {m.event_id} {m.as_observation_for(agent.name)[:300]}" for m in unpresented)
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
    public_entities = {key: state.entity_states.get(key, {})
                       for key, spec in getattr(state.world_spec, 'entities', {}).items()
                       if isinstance(spec, dict) and can_access(spec.get('visibility'), public)}

    def effect_visible(op):
        if op.get('op') in public_ops:
            return True
        if op.get('op') in {'set_entity', 'settle_votes'}:
            return op.get('target') in public_entities and op.get('key') in public_entities[op['target']]
        if op.get('op') in {'set_world', 'increment_world'}:
            field = state.state_schema.get(op.get('key'))
            return bool(field and can_access(field.visibility, public))
        return False
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
        "public_entity_states_now": public_entities,
        "state_instruction": "这些是当前已提交状态；开场位置、准备动作和台词主张不能把它们重置。",
        "public_vote_readback": vote_readback,
        "cast": [a.name for a in state.agents.values()],
        "public_active_cast": active_names,
        "public_active_count": len(active_names) if active_names is not None else None,
        "director_mutable_fields": [key for key, field in state.state_schema.items()
                                    if "director" in field.mutable_by and "public" in field.visibility
                                    and key not in {"current_phase", "round_number"}],
        "committed_events": [{"event_id": m.event_id,
                              "action": ("系统安全推进，非角色回应" if m.intent.get("generation_fallback") else m.action),
                              "committed_public_effects": [op for op in m.state_patch if effect_visible(op)],
                              "public_changes": m.public_changes} for m in events],
    }, ensure_ascii=False)
