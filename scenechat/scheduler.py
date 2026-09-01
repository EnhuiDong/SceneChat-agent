from __future__ import annotations

from dataclasses import dataclass

from .models import AgentState, SimulationState


@dataclass(frozen=True)
class SchedulerDecision:
    kind: str
    actor_name: str = ""
    reason: str = ""


class SimulationScheduler:
    """Choose from public eligibility, phase and location state only."""

    def decide(self, state: SimulationState) -> SchedulerDecision:
        if state.pending_events and state.scheduler_strategy == "event_first":
            return SchedulerDecision("event", reason="存在待处理公共环境事件")

        phase = state.phase_specs.get(state.current_phase)
        if phase is not None and getattr(phase, "event_only", False):
            return SchedulerDecision("event", reason="当前为环境事件阶段")
        strategy = getattr(phase, "scheduler", "") or state.scheduler_strategy
        eligible = self._eligible_for_phase(state, phase)
        if not eligible:
            return SchedulerDecision("narration", reason="当前阶段没有可行动角色")

        response_candidates = [agent for agent in eligible if agent.pending_intents]
        if response_candidates:
            actor = max(response_candidates, key=self._response_priority)
            pending = max(actor.pending_intents, key=self._pending_priority)
            return SchedulerDecision(
                "agent",
                actor.name,
                f"优先回应 {pending.get('speaker') or '上一位角色'} 的直接"
                f"{pending.get('move') or '发言'}",
            )
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
        return SchedulerDecision("agent", actor.name, f"使用 {strategy} 调度")

    @staticmethod
    def _eligible_for_phase(state: SimulationState, phase) -> list[AgentState]:
        roles = list(getattr(phase, "actor_roles", []) or [])
        unacted = [
            agent
            for agent in state.agents.values()
            if agent.eligible
            and (not roles or agent.role in roles)
            and agent.name not in state.phase_action_log
        ]
        if unacted:
            return unacted
        # Free-running phases may continue round-robin. Structured phases wait
        # for their resolver transition instead of selecting an invalid actor.
        if phase is None:
            return [agent for agent in state.agents.values() if agent.eligible]
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
