from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import AgentState, SimulationState
from .role_selectors import matches_role


@dataclass(frozen=True)
class SchedulerDecision:
    kind: str
    actor_name: str = ""
    reason: str = ""
    thread_id: str = ""
    obligation_id: str = ""
    source_event_id: str = ""
    phase_exit_rule_id: str = ""


class SimulationScheduler:
    """Choose from public eligibility, phase and location state only."""

    def decide(self, state: SimulationState) -> SchedulerDecision:
        if state.pending_events and state.scheduler_strategy == "event_first":
            return self._record(state, SchedulerDecision("event", reason="存在待处理公共环境事件"))

        phase = state.phase_specs.get(state.current_phase)
        if phase is not None and getattr(phase, "event_only", False):
            if not phase.next_phase and state.history:
                previous = state.history[-1]
                if previous.kind == "narration" and previous.intent.get("narration_phase") == phase.name:
                    return self._record(state, SchedulerDecision(
                        "blocked", reason=f"终局阶段“{phase.name}”已执行环境事件，但结束条件仍未满足"
                    ))
            return self._record(state, SchedulerDecision("event", reason="当前为环境事件阶段"))
        strategy = getattr(phase, "scheduler", "") or state.scheduler_strategy
        eligible = self._eligible_for_phase(state, phase)
        if not eligible:
            return self._record(state, SchedulerDecision(
                "blocked", reason=f"阶段“{state.current_phase}”没有符合规则的可行动角色，或阶段行动已耗尽但未完成切换"
            ))

        exit_rule = self._due_phase_exit(state, phase, eligible)
        if exit_rule is not None:
            from .visibility import ViewerContext, can_access
            actor = self._round_robin_actor(
                state, [agent for agent in eligible
                        if matches_role(agent.role, exit_rule.allowed_roles)
                        and can_access(exit_rule.visibility, ViewerContext(
                            name=agent.name, role=agent.role, location=agent.current_location))]
            )
            return self._record(state, SchedulerDecision(
                "agent", actor.name,
                f"本阶段已充分讨论；由角色按规则 {exit_rule.id} 执行阶段切换，避免重复争论",
                phase_exit_rule_id=exit_rule.id,
            ))

        # Model-supplied urgency cannot starve the rest of an open phase.
        recent_actors = [m.speaker for m in state.history if m.speaker in {a.name for a in eligible}]
        window = max(4, len(eligible) * 2)
        if len(recent_actors) >= window:
            overdue = [a for a in eligible if a.name not in recent_actors[-window:]]
            if overdue:
                actor = max(overdue, key=lambda a: self._waiting_turns(state, a))
                return self._record(state, SchedulerDecision("agent", actor.name, "避免连续插话使其他可行动角色长期无回合"))

        response_candidates = [agent for agent in eligible if agent.pending_intents]
        if response_candidates:
            def response_score(agent, pending):
                urgency, created = self._pending_priority(pending)
                age = max(0, state.turn_count - created)
                return ((1 if pending.get("address_scope", "specific") == "specific" else 0)
                        + min(urgency, 1.0) * 0.4 + age / max(4, len(eligible))
                        + self._waiting_turns(state, agent) / max(4, len(eligible)))
            actor = max(response_candidates, key=lambda a: max(response_score(a, p) for p in a.pending_intents))
            pending = max(actor.pending_intents, key=lambda p: response_score(actor, p))
            return self._record(state, SchedulerDecision(
                "agent",
                actor.name,
                f"优先回应 {pending.get('speaker') or '上一位角色'} 的直接"
                f"{pending.get('move') or '发言'}",
                str(pending.get("thread_id") or ""),
                str(pending.get("obligation_id") or ""),
                str(pending.get("event_id") or ""),
            ))
        # Once all addressees have replied but the request is still blocked,
        # return control to its owner once. This is a choice point, not an
        # invitation to make every participant agree again.
        from .agenda import relevant_tasks
        followups = [
            (agent, task) for agent in eligible
            for task in relevant_tasks(state, agent, limit=8)
            if task.owner == agent.name and task.status == "blocked"
            and task.owner_reviewed_at_turn < task.updated_at_turn
        ]
        if followups:
            actor, task = max(followups, key=lambda pair: pair[1].updated_at_turn)
            return self._record(state, SchedulerDecision(
                "agent", actor.name,
                "已有回应但事项尚未解决；请求者选择核验、改变条件、执行替代方案或搁置",
                task.thread_id, "", task.source_event_id,
            ))
        opportunities_by_agent = {
            agent.name: [
                item for item in agent.conversation_opportunities
                if self._pending_priority(item)[1] >= state.turn_count - 4
            ]
            for agent in eligible
        }
        opportunity_candidates = [
            agent for agent in eligible if opportunities_by_agent[agent.name]
        ]
        if opportunity_candidates:
            recent_speakers = [
                message.speaker for message in state.history[-2:]
                if message.speaker in state.agents
            ]
            fresh_candidates = [
                agent for agent in opportunity_candidates
                if agent.name not in recent_speakers
            ] or opportunity_candidates
            actor = max(
                fresh_candidates,
                key=lambda item: self._opportunity_priority(
                    opportunities_by_agent[item.name]
                ),
            )
            opportunity = max(
                opportunities_by_agent[actor.name],
                key=self._pending_priority,
            )
            return self._record(state, SchedulerDecision(
                "agent",
                actor.name,
                f"{opportunity.get('speaker') or '上一位角色'}提及了该角色，"
                "允许其按相关性选择插话",
                str(opportunity.get("thread_id") or ""),
                "",
                str(opportunity.get("event_id") or ""),
            ))
        if strategy == "initiative":
            actor = sorted(eligible, key=lambda item: (-item.initiative, item.name))[0]
        elif strategy == "urgency_director":
            actor = sorted(
                eligible,
                key=lambda item: (-self._public_urgency(item), item.name),
            )[0]
        elif strategy == "phase_order":
            actor = self._phase_order_actor(state, phase, eligible)
        else:
            actor = self._round_robin_actor(state, eligible)
        return self._record(
            state, SchedulerDecision("agent", actor.name, f"使用 {strategy} 调度")
        )

    @staticmethod
    def _waiting_turns(state: SimulationState, agent: AgentState) -> int:
        count = 0
        for message in reversed(state.history):
            if message.speaker == agent.name:
                break
            if message.speaker in state.agents:
                count += 1
        return count

    @staticmethod
    def _record(state: SimulationState, decision: SchedulerDecision) -> SchedulerDecision:
        if decision.kind == "agent" and decision.actor_name in state.agent_order:
            # Direct replies consume a real actor turn too. Otherwise, after a
            # broadcast response round the fallback cursor replays that round.
            state._scheduler_index = state.agent_order.index(decision.actor_name) + 1
        state.last_scheduler_decision = {
            "kind": decision.kind,
            "actor_name": decision.actor_name,
            "reason": decision.reason,
            "thread_id": decision.thread_id,
            "obligation_id": decision.obligation_id,
            "source_event_id": decision.source_event_id,
            "phase_exit_rule_id": decision.phase_exit_rule_id,
            "at_turn": state.turn_count,
        }
        return decision

    @staticmethod
    def _due_phase_exit(state: SimulationState, phase, eligible: list[AgentState]):
        if (phase is None or phase.event_only or phase.advance_when != "manual"
                or not eligible):
            return None
        from .visibility import ViewerContext, can_access

        def actors_for(rule):
            return [agent for agent in eligible
                    if matches_role(agent.role, rule.allowed_roles)
                    and can_access(rule.visibility, ViewerContext(
                        name=agent.name, role=agent.role, location=agent.current_location))]

        exits = [
            rule for rule in state.rules
            if (not rule.phases or phase.name in rule.phases)
            and (getattr(state.world_spec, 'execution_version', 1) != 2
                 or rule.effect_mode in {'rule', 'stack'})
            and rule.action_type in phase.allowed_action_types
            and rule.target_scope in {"none", "self"}
            and any(effect.op == "set_phase" and effect.value in state.phase_specs
                    and effect.value != phase.name
                    for effect in rule.effects)
            and actors_for(rule)
        ]
        if phase.next_phase:
            exits = [rule for rule in exits if any(effect.op == 'set_phase' and effect.value == phase.next_phase
                                                   for effect in rule.effects)]
        else:
            # A sole explicit route to a structured decision is still real
            # even if the generated phase omitted next_phase. Do not invent
            # an exit, choose a branch, or time-limit free conversation.
            destinations = {effect.value for rule in exits for effect in rule.effects if effect.op == 'set_phase'}
            if len(destinations) != 1:
                return None
            destination = state.phase_specs.get(next(iter(destinations)))
            if (destination is None or destination.event_only
                    or destination.advance_when not in {'all_active_voted', 'all_eligible_acted'}
                    or not destination.allowed_action_types
                    or set(destination.allowed_action_types).intersection({'speak', 'act', 'observe', 'pass'})):
                return None
        if not exits:
            return None
        names = {agent.name for agent in eligible}
        phase_start = 0
        for index, message in enumerate(state.history):
            if any(operation.get("op") == "set_phase" and operation.get("value") == phase.name
                   for operation in message.state_patch):
                phase_start = index + 1
            elif message.state_updates.get("current_phase") == phase.name:
                phase_start = index + 1
        actor_turns = sum(message.speaker in names for message in state.history[phase_start:])
        pace = getattr(getattr(state, "arc_state", None), "pace", 50)
        cycles = 3 if pace <= 30 else 1 if pace > 70 else 2
        if actor_turns < len(eligible) * cycles or not names.issubset(state.phase_action_log):
            return None
        return max(exits, key=lambda rule: rule.priority)

    @staticmethod
    def _eligible_for_phase(state: SimulationState, phase) -> list[AgentState]:
        roles = list(getattr(phase, "actor_roles", []) or [])
        eligible = [
            agent
            for agent in state.agents.values()
            if agent.eligible
            and matches_role(agent.role, roles)
        ]
        # A manual phase is deliberately open-ended.  phase_action_log tracks
        # who acted in the current structured cycle, but must never exhaust an
        # ongoing discussion, negotiation, exploration, or combat phase.
        if phase is None or getattr(phase, "advance_when", "") == "manual":
            return eligible
        if (getattr(phase, "advance_when", "") == "all_eligible_acted"
                and not phase.next_phase
                and any(not rule.phases or phase.name in rule.phases
                        for rule in state.termination_rules)):
            # No transition is possible. A terminal task phase remains open
            # until its actual termination predicate is met, not merely until
            # everyone has spoken. Keep logs intact and invent no completion.
            return eligible
        acted, previous_turns, revisited = state.phase_qualifying_participation()
        unacted = [agent for agent in eligible if agent.name not in acted]
        if unacted:
            return unacted
        if getattr(phase, "advance_when", "") == "all_eligible_acted":
            required = state.phase_required_actor_turns(phase, len(eligible), revisited=revisited)
            if previous_turns < required:
                return eligible
        # Structured phases wait for their resolver transition instead of
        # selecting an actor twice inside the same cycle.
        return []

    @staticmethod
    def _round_robin_actor(state: SimulationState, eligible: list[AgentState]) -> AgentState:
        names = {agent.name for agent in eligible}
        for _ in range(len(state.agent_order)):
            name = state.agent_order[state._scheduler_index % len(state.agent_order)]
            state._scheduler_index += 1
            if name in names:
                return state.agents[name]
        return eligible[0]

    @staticmethod
    def _phase_order_actor(state: SimulationState, phase, eligible: list[AgentState]) -> AgentState:
        roles = list(getattr(phase, "actor_roles", []) or [])
        if roles:
            by_role = {role: index for index, role in enumerate(roles)}
            return sorted(eligible, key=lambda item: (by_role.get(item.role, len(roles)), item.name))[0]
        return SimulationScheduler._round_robin_actor(state, eligible)

    @staticmethod
    def _public_urgency(agent: AgentState) -> int:
        value = agent.resources.get("public_urgency", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _pending_priority(item: dict) -> tuple[float, int]:
        try:
            urgency = float(item.get("urgency", 0) or 0)
        except (TypeError, ValueError):
            urgency = 0.0
        try:
            created_at = int(item.get("created_at_turn", 0) or 0)
        except (TypeError, ValueError):
            created_at = 0
        return urgency, created_at

    @classmethod
    def _response_priority(cls, agent: AgentState) -> tuple[float, int, str]:
        urgency, created_at = max(
            (cls._pending_priority(item) for item in agent.pending_intents),
            default=(0.0, 0),
        )
        return urgency, created_at, agent.name

    @classmethod
    def _opportunity_priority(
        cls,
        opportunities: list[dict[str, Any]],
    ) -> tuple[float, int]:
        urgency, created_at = max(
            (cls._pending_priority(item) for item in opportunities),
            default=(0.0, 0),
        )
        return urgency, created_at
