"""Evidence-backed, visibility-aware tasks derived from committed interactions.

An agenda item tracks the *work of obtaining a response*, not the truth of a
character's answer. Dialogue never commits a world fact through this reducer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgendaItem:
    id: str
    source_event_id: str
    thread_id: str
    title: str
    owner: str
    targets: list[str] = field(default_factory=list)
    visibility: list[str] = field(default_factory=lambda: ["public"])
    location: str = ""
    phase: str = ""
    status: str = "active"  # active, blocked, completed, abandoned
    next_action: str = ""
    evidence_event_ids: list[str] = field(default_factory=list)
    created_at_turn: int = 0
    updated_at_turn: int = 0
    owner_reviewed_at_turn: int = 0


def agenda_from_dict(data: dict) -> AgendaItem:
    """Restore only known, bounded fields from an owner export."""
    return AgendaItem(
        id=str(data.get("id") or "")[:100],
        source_event_id=str(data.get("source_event_id") or "")[:100],
        thread_id=str(data.get("thread_id") or "")[:100],
        title=str(data.get("title") or "")[:240],
        owner=str(data.get("owner") or "")[:100],
        targets=[str(x)[:100] for x in (data.get("targets") or [])[:16]],
        visibility=[str(x)[:100] for x in (data.get("visibility") or ["public"])[:8]],
        location=str(data.get("location") or "")[:100],
        phase=str(data.get("phase") or "")[:100],
        status=str(data.get("status") or "active") if data.get("status") in {"active", "blocked", "completed", "abandoned"} else "active",
        next_action=str(data.get("next_action") or "")[:240],
        evidence_event_ids=[str(x)[:100] for x in (data.get("evidence_event_ids") or [])[-20:]],
        created_at_turn=max(0, int(data.get("created_at_turn") or 0)),
        updated_at_turn=max(0, int(data.get("updated_at_turn") or 0)),
        owner_reviewed_at_turn=max(0, int(data.get("owner_reviewed_at_turn") or 0)),
    )


def visible_to(state, item: AgendaItem, agent) -> bool:
    source = next((m for m in reversed(state.history) if m.event_id == item.source_event_id), None)
    return bool(source and state._agent_can_observe(agent, source))


def relevant_tasks(state, agent, limit: int = 3) -> list[AgendaItem]:
    items = [
        item for item in state.agenda.values()
        if item.status in {"active", "blocked"}
        and item.phase == state.current_phase
        and (agent.name == item.owner or agent.name in item.targets)
        and visible_to(state, item, agent)
    ]
    return sorted(items, key=lambda item: item.updated_at_turn, reverse=True)[:limit]


def update_agenda(state, message) -> None:
    """Reduce committed obligations; reject model-authored task completion claims."""
    if message.speaker in state.agents and not message.intent.get("generation_fallback"):
        move = str(message.intent.get("conversation_move") or "")
        if move in {"question", "request", "challenge"}:
            thread = state.thread_for_event(message.event_id)
            obligations = [o for o in thread.obligations if o.source_event_id == message.event_id] if thread else []
            if obligations:
                # A second request on one thread supersedes a stalled collection,
                # but does not erase its evidence or pretend it was completed.
                parent_event = str(message.intent.get("reply_to_event_id") or "")
                for old in state.agenda.values():
                    if (old.thread_id == thread.id and old.status == "blocked"
                            and parent_event
                            and parent_event in old.evidence_event_ids):
                        old.status = "abandoned"
                        old.next_action = "已由新的具体请求取代"
                        old.updated_at_turn = message.turn
                        old.evidence_event_ids.append(message.event_id)
                item = AgendaItem(
                    id=f"task-{message.event_id[:24]}",
                    source_event_id=message.event_id,
                    thread_id=thread.id,
                    title=(message.speech or message.action).strip()[:240],
                    owner=message.speaker,
                    targets=list(dict.fromkeys(o.target for o in obligations)),
                    visibility=list(message.scopes),
                    location=message.location,
                    phase=state.current_phase,
                    next_action="等待指定对象逐一作出实质回应；回应不等于事实已核验",
                    evidence_event_ids=[message.event_id],
                    created_at_turn=message.turn,
                    updated_at_turn=message.turn,
                )
                state.agenda[item.id] = item

    for item in state.agenda.values():
        if item.status not in {"active", "blocked"}:
            continue
        if item.phase != state.current_phase:
            item.status = "abandoned"
            item.next_action = "场景阶段已改变；原请求不再自动调度"
            item.updated_at_turn = message.turn
            item.evidence_event_ids.append(message.event_id)
            continue
        thread = state.conversation_threads.get(item.thread_id)
        obligations = [o for o in thread.obligations if o.source_event_id == item.source_event_id] if thread else []
        if not obligations:
            item.status = "abandoned"
            item.next_action = "原始回应义务已不在当前追踪窗口内"
            item.updated_at_turn = message.turn
            continue
        evidence = [o.resolution_event_id for o in obligations if o.resolution_event_id]
        for event_id in evidence:
            if event_id not in item.evidence_event_ids:
                item.evidence_event_ids.append(event_id)
                item.updated_at_turn = message.turn
        item.evidence_event_ids = item.evidence_event_ids[-20:]
        open_targets = [o.target for o in obligations if o.status == "open"]
        if open_targets:
            item.status = "active"
            item.next_action = "等待 " + "、".join(open_targets[:8]) + " 回应；可回答、拒绝或说明不知道"
        elif any(o.status in {"responded", "expired"} for o in obligations):
            if item.status != "blocked":
                item.updated_at_turn = message.turn
            item.status = "blocked"
            item.next_action = "请求者根据已有回应选择核验、改变条件、执行替代行动或明确搁置；不要原样追问"
        else:
            item.status = "completed"
            item.next_action = "已收到本次请求的回应；回答内容仍只是角色主张"
            item.updated_at_turn = message.turn
        if message.speaker == item.owner and message.turn > item.created_at_turn and item.status == "blocked":
            item.owner_reviewed_at_turn = message.turn
    # Bound persistence and prompt cost; completed history is still in events.
    if len(state.agenda) > 80:
        removable = [key for key, item in state.agenda.items() if item.status in {"completed", "abandoned"}]
        for key in removable[:len(state.agenda) - 80]:
            state.agenda.pop(key, None)
