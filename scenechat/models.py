from dataclasses import dataclass, field
from typing import Any, Dict, List
import uuid

from .visibility import ViewerContext, can_access, normalize_scopes


MAX_AGENT_MEMORY = 20
MAX_AGENT_OBSERVATIONS = 30


@dataclass
class Message:
    speaker: str
    action: str
    speech: str
    turn: int
    kind: str = "dialogue"
    visibility: str = "public"
    state_updates: Dict[str, str] = field(default_factory=dict)
    memory: str = ""
    relationship_updates: Dict[str, str] = field(default_factory=dict)
    end_signal: bool = False
    end_reason: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    visibility_scopes: List[str] = field(default_factory=list)
    location: str = ""
    participants: List[str] = field(default_factory=list)
    individual_observations: Dict[str, str] = field(default_factory=dict)
    authoritative: bool = False
    state_patch: List[Dict[str, Any]] = field(default_factory=list)
    intent: Dict[str, Any] = field(default_factory=dict)

    def as_observation(self) -> str:
        return f"[{self.turn}] {self.speaker}：{self.action} {self.speech}".strip()

    def as_observation_for(self, agent_name: str) -> str:
        content = self.individual_observations.get(agent_name)
        if content:
            return f"[{self.turn}] {content}".strip()
        return self.as_observation()

    @property
    def scopes(self) -> List[str]:
        return normalize_scopes(self.visibility_scopes or self.visibility)


@dataclass
class MemoryRecord:
    event_id: str
    content: str
    source: str = "observation"
    visibility: List[str] = field(default_factory=lambda: ["agent_private"])
    confidence: float = 1.0
    importance: int = 1
    created_at_turn: int = 0


@dataclass
class AbilityState:
    id: str
    name: str
    action_type: str
    description: str = ""
    phases: List[str] = field(default_factory=list)
    uses_remaining: int | None = None
    target_scope: str = "same_location"
    effects: List[Dict[str, Any]] = field(default_factory=list)
    visibility: List[str] = field(default_factory=lambda: ["public"])

    @property
    def available(self) -> bool:
        return self.uses_remaining is None or self.uses_remaining > 0


@dataclass
class AgentState:
    """Private state and observable knowledge belonging to a single character."""

    name: str
    profile: str
    public_profile: str
    goals: List[str] = field(default_factory=list)
    private_memory: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)
    role: str = ""
    abilities: List[str] = field(default_factory=list)
    id: str = ""
    faction: str = ""
    current_location: str = ""
    active: bool = True
    alive: bool = True
    resources: Dict[str, Any] = field(default_factory=dict)
    ability_states: Dict[str, AbilityState] = field(default_factory=dict)
    goal_status: Dict[str, str] = field(default_factory=dict)
    known_facts: Dict[str, str] = field(default_factory=dict)
    false_beliefs: List[str] = field(default_factory=list)
    current_emotion: str = "平静"
    pending_intents: List[Dict[str, Any]] = field(default_factory=list)
    memories: List[MemoryRecord] = field(default_factory=list)
    initiative: int = 0

    def observe(self, message: Message) -> None:
        observation = message.as_observation_for(self.name)
        self.observations.append(observation)
        self.observations = self.observations[-MAX_AGENT_OBSERVATIONS:]

        self.memories.append(
            MemoryRecord(
                event_id=message.event_id,
                content=observation,
                source="direct_observation",
                visibility=[f"agent:{self.name}"],
                created_at_turn=message.turn,
            )
        )
        self.memories = self.memories[-MAX_AGENT_MEMORY:]

        if message.speaker == self.name:
            self.remember(f"我曾经采取行动：{message.action}；并说：{message.speech}")

    def remember(self, memory: str) -> None:
        memory = memory.strip()
        if not memory:
            return
        self.private_memory.append(memory)
        self.private_memory = self.private_memory[-MAX_AGENT_MEMORY:]
        self.memories.append(
            MemoryRecord(
                event_id=f"memory-{uuid.uuid4().hex}",
                content=memory,
                source="private_reflection",
                visibility=[f"agent:{self.name}"],
            )
        )
        self.memories = self.memories[-MAX_AGENT_MEMORY:]

    def ability_by_reference(self, reference: str) -> AbilityState | None:
        value = reference.strip()
        if not value:
            return None
        if value in self.ability_states:
            return self.ability_states[value]
        return next(
            (ability for ability in self.ability_states.values() if ability.name == value),
            None,
        )

    @property
    def eligible(self) -> bool:
        return self.active and self.alive

    def recent_observations(self, limit: int = 15) -> str:
        observations = self.observations[-limit:]
        return "\n".join(observations) if observations else "尚未观察到其他行动。"

    def recent_private_memory(self, limit: int = 10) -> str:
        memories = self.private_memory[-limit:]
        return "\n".join(memories) if memories else "暂无额外私人记忆。"


class SimulationState:
    def __init__(self, scene: str, agents: List[AgentState], world_spec: Any = None):
        if not agents:
            raise ValueError("角色设定中没有解析出有效角色")
        self.scene = scene
        self.agents: Dict[str, AgentState] = {agent.name: agent for agent in agents}
        self.agent_order = list(self.agents)
        self.history: List[Message] = []
        self.turn_count = 0
        self.agent_turn_count = 0
        self.narration_count = 0
        self._scheduler_index = 0
        self.world_spec = world_spec
        self.phase_sequence = list(getattr(world_spec, "phases", []) or [])
        self.current_phase = self.phase_sequence[0] if self.phase_sequence else "自由推进"
        self.public_rules = list(getattr(world_spec, "public_rules", []) or [])
        self.world_state: Dict[str, Any] = dict(
            getattr(world_spec, "initial_state", {}) or {}
        )
        self.termination_conditions = list(
            getattr(world_spec, "termination_conditions", []) or []
        )
        self.ended = False
        self.end_reason = ""
        self.end_kind = ""
        self.winner = ""
        self.scheduler_strategy = getattr(world_spec, "scheduler", "round_robin") or "round_robin"
        self.phase_specs = {
            phase.name: phase for phase in list(getattr(world_spec, "phase_specs", []) or [])
        }
        self.rules = list(getattr(world_spec, "rules", []) or [])
        self.state_schema = dict(getattr(world_spec, "state_schema", {}) or {})
        self.termination_rules = list(getattr(world_spec, "termination_rules", []) or [])
        self.facts = {fact.id: fact for fact in list(getattr(world_spec, "facts", []) or [])}
        self.locations = list(getattr(world_spec, "locations", []) or [])
        self.phase_action_log: set[str] = set()
        self.votes: Dict[str, str] = {}
        self.pending_events: List[Dict[str, Any]] = []
        self.protected_agents: set[str] = set()
        self.failed_generation_count = 0
        self.run_status = "running"

        default_location = self.locations[0] if self.locations else scene
        for agent in self.agents.values():
            if not agent.current_location:
                agent.current_location = default_location
            for fact in self.facts.values():
                viewer = ViewerContext(
                    name=agent.name,
                    role=agent.role,
                    location=agent.current_location,
                )
                if fact.id in agent.known_facts or can_access(fact.visibility, viewer):
                    agent.known_facts[fact.id] = fact.content

    def next_agent(self) -> AgentState:
        """Use a transparent round-robin scheduler without private omniscience."""
        for _ in range(len(self.agent_order)):
            name = self.agent_order[self._scheduler_index % len(self.agent_order)]
            self._scheduler_index += 1
            agent = self.agents[name]
            if agent.eligible:
                return agent
        raise RuntimeError("当前没有可行动角色")

    def add_message(self, msg: Message) -> None:
        self.history.append(msg)
        self.turn_count += 1
        if msg.kind == "narration":
            self.narration_count += 1
        else:
            self.agent_turn_count += 1

        # Observation is recipient-specific. A public local event reaches only
        # people at that location; role/agent scopes and explicit private
        # observations are checked in code rather than entrusted to the model.
        for agent in self.agents.values():
            if self._agent_can_observe(agent, msg):
                agent.observe(msg)

        active_agent = self.agents.get(msg.speaker)
        if active_agent is not None:
            if msg.memory:
                active_agent.remember(msg.memory)
            for target, update in msg.relationship_updates.items():
                if target in self.agents and target != active_agent.name and update.strip():
                    active_agent.relationships[target] = update.strip()

        # Dialogue describes an agent's attempted action; it is not an
        # authoritative rule resolution. Only an observable narrator event may
        # commit public state so a character cannot grant itself an ability,
        # force another character's decision, or declare its own victory.
        is_public_resolution = msg.visibility == "public" and msg.kind == "narration"
        if msg.speaker in self.agents and msg.authoritative:
            self.phase_action_log.add(msg.speaker)
        if msg.authoritative:
            self.apply_state_patch(msg.state_patch)
        elif is_public_resolution:
            self.apply_state_updates(msg.state_updates, allow_phase_change=True)
        if (msg.authoritative or is_public_resolution) and msg.end_signal:
            self.ended = True
            self.end_reason = msg.end_reason.strip() or "场景结束条件已经满足。"
            self.end_kind = "natural_end"
            self.run_status = "ended"

        self.evaluate_termination()

    def _agent_can_observe(self, agent: AgentState, msg: Message) -> bool:
        if agent.name in msg.individual_observations:
            return True
        viewer = ViewerContext(
            name=agent.name,
            role=agent.role,
            location=agent.current_location,
        )
        if not can_access(msg.scopes, viewer):
            return False
        if msg.participants and agent.name not in msg.participants:
            return False
        if msg.location and msg.location != agent.current_location:
            return False
        return True

    def apply_state_patch(self, operations: List[Dict[str, Any]]) -> None:
        """Apply resolver-created operations through a strict operation whitelist."""

        for operation in list(operations or [])[:50]:
            if not isinstance(operation, dict):
                continue
            op = str(operation.get("op") or "").strip()
            key = str(operation.get("key") or "").strip()
            target_name = str(operation.get("target") or "").strip()
            value = operation.get("value")
            agent = self.agents.get(target_name)
            if op == "set_world" and self._valid_world_value(key, value):
                self.world_state[key] = value
            elif op == "increment_world" and key:
                try:
                    self.world_state[key] = int(self.world_state.get(key, 0)) + int(
                        operation.get("amount", 1)
                    )
                except (TypeError, ValueError):
                    continue
            elif op == "move_agent" and agent is not None:
                location = str(value or "").strip()
                if location and (not self.locations or location in self.locations):
                    agent.current_location = location
            elif op == "set_agent_status" and agent is not None and key in {"active", "alive"}:
                setattr(agent, key, bool(value))
                if key == "alive" and not bool(value):
                    agent.active = False
            elif op in {"set_resource", "consume_resource"} and agent is not None and key:
                if op == "set_resource":
                    agent.resources[key] = value
                else:
                    try:
                        current = int(agent.resources.get(key, 0))
                        agent.resources[key] = max(0, current - int(operation.get("amount", 1)))
                    except (TypeError, ValueError):
                        continue
            elif op == "consume_ability" and agent is not None and key:
                ability = agent.ability_states.get(key)
                if ability is not None and ability.uses_remaining is not None:
                    try:
                        amount = max(1, int(operation.get("amount", 1)))
                    except (TypeError, ValueError):
                        amount = 1
                    ability.uses_remaining = max(0, ability.uses_remaining - amount)
            elif op == "protect_agent" and target_name in self.agents:
                self.protected_agents.add(target_name)
            elif op == "clear_protections":
                self.protected_agents.clear()
            elif op == "set_goal_status" and agent is not None and key:
                agent.goal_status[key] = str(value)
            elif op == "set_relationship" and agent is not None and key in self.agents:
                agent.relationships[key] = str(value)
            elif op == "record_vote" and target_name in self.agents:
                voter = str(operation.get("actor") or "").strip()
                if voter in self.agents and self.agents[voter].eligible:
                    self.votes[voter] = target_name
            elif op == "clear_votes":
                self.votes.clear()
            elif op == "set_phase":
                self.advance_phase(str(value or "").strip())
            elif op == "add_known_fact" and agent is not None and key:
                agent.known_facts[key] = str(value)

    def _valid_world_value(self, key: str, value: Any) -> bool:
        if not key or key.startswith("_"):
            return False
        field_spec = self.state_schema.get(key)
        if field_spec is None:
            return not self.state_schema
        allowed_values = list(getattr(field_spec, "allowed_values", []) or [])
        if allowed_values and str(value) not in allowed_values:
            return False
        value_type = getattr(field_spec, "value_type", "string")
        return (
            (value_type == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (value_type == "boolean" and isinstance(value, bool))
            or (value_type in {"string", "enum"} and isinstance(value, str))
            or value_type == "any"
        )

    def advance_phase(self, requested_phase: str = "") -> None:
        if not self.phase_sequence:
            return
        if requested_phase and requested_phase in self.phase_sequence:
            next_phase = requested_phase
        else:
            current_index = self.phase_sequence.index(self.current_phase)
            next_phase = self.phase_sequence[(current_index + 1) % len(self.phase_sequence)]
        self.current_phase = next_phase
        self.phase_action_log.clear()
        self.votes.clear()
        self.protected_agents.clear()

    def evaluate_termination(self) -> None:
        if self.ended:
            return
        for rule in self.termination_rules:
            matched = self._termination_matches(rule)
            if matched:
                self.ended = True
                self.end_kind = "natural_end"
                self.run_status = "ended"
                self.end_reason = getattr(rule, "description", "") or "场景结束条件已经满足。"
                self.winner = getattr(rule, "winner", "")
                return

    def _termination_matches(self, rule: Any) -> bool:
        phases = list(getattr(rule, "phases", []) or [])
        if phases and self.current_phase not in phases:
            return False
        kind = getattr(rule, "kind", "manual")
        conditions = list(getattr(rule, "conditions", []) or [])
        if kind == "all_of":
            return bool(conditions) and all(
                self._termination_matches(condition) for condition in conditions
            )
        if kind == "any_of":
            return bool(conditions) and any(
                self._termination_matches(condition) for condition in conditions
            )
        faction = getattr(rule, "faction", "")
        alive = [agent for agent in self.agents.values() if agent.eligible]
        if kind == "faction_eliminated" and faction:
            return not any(agent.faction == faction for agent in alive)
        if kind == "faction_parity" and faction:
            faction_count = sum(agent.faction == faction for agent in alive)
            return faction_count > 0 and faction_count >= len(alive) - faction_count
        if kind == "world_equals":
            return self.world_state.get(getattr(rule, "key", "")) == getattr(rule, "value", None)
        if kind == "all_goals_completed":
            return bool(alive) and all(
                agent.goal_status
                and all(value == "completed" for value in agent.goal_status.values())
                for agent in alive
            )
        if kind == "all_active_at_location":
            location = getattr(rule, "location", "")
            return bool(alive) and bool(location) and all(
                agent.current_location == location for agent in alive
            )
        return False

    @property
    def can_continue(self) -> bool:
        return not self.ended and self.run_status == "running"

    def record_generation_failure(self) -> None:
        self.failed_generation_count += 1
        if self.failed_generation_count >= 3:
            self.run_status = "blocked"
            self.end_kind = "blocked"
            self.end_reason = "模型连续三次未能产生合法行动，模拟已停止以保护状态一致性。"

    def record_generation_success(self) -> None:
        self.failed_generation_count = 0

    def apply_state_updates(
        self,
        updates: Dict[str, str],
        *,
        allow_phase_change: bool = False,
    ) -> None:
        """Apply a small, text-only state patch produced by a resolved event.

        The whitelist prevents a model response from mutating Python object
        attributes. A phase change is accepted only when it names a declared
        phase; all other entries remain ordinary world-state variables.
        """

        if not isinstance(updates, dict):
            return
        for raw_key, raw_value in list(updates.items())[:30]:
            key = str(raw_key).strip()
            value = str(raw_value).strip()
            if not key or not value:
                continue
            if key == "phase":
                if allow_phase_change and value in self.phase_sequence:
                    self.current_phase = value
                continue
            if key.startswith("_"):
                continue
            self.world_state[key[:80]] = value[:1000]

    def public_state_summary(self) -> str:
        variables = "\n".join(
            f"- {key}：{value}" for key, value in self.public_world_state().items()
        )

    def public_world_state(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        public_viewer = ViewerContext()
        for key, value in self.world_state.items():
            field_spec = self.state_schema.get(key)
            scopes = getattr(field_spec, "visibility", ["public"])
            if can_access(scopes, public_viewer):
                result[key] = value
        return result
        conditions = "\n".join(
            f"- {condition}" for condition in self.termination_conditions
        )
        rules = "\n".join(f"- {rule}" for rule in self.public_rules)
        return (
            f"当前阶段：{self.current_phase}\n"
            f"必须遵守的公共规则：\n{rules or '- 依据现实常识与场景设定'}\n"
            f"公共状态：\n{variables or '- 暂无额外状态变量'}\n"
            f"结束条件：\n{conditions or '- 尚未声明'}"
        )

    def state_summary_for(self, agent: AgentState) -> str:
        viewer = ViewerContext(
            name=agent.name,
            role=agent.role,
            location=agent.current_location,
        )
        visible_variables = []
        for key, value in self.world_state.items():
            field_spec = self.state_schema.get(key)
            scopes = getattr(field_spec, "visibility", ["public"])
            if can_access(scopes, viewer):
                visible_variables.append(f"- {key}：{value}")
        rules = "\n".join(f"- {rule}" for rule in self.public_rules)
        conditions = "\n".join(f"- {condition}" for condition in self.termination_conditions)
        phase = self.phase_specs.get(self.current_phase)
        allowed_actions = list(getattr(phase, "allowed_action_types", []) or [])
        action_line = "、".join(allowed_actions) if allowed_actions else "自由行动"
        return (
            f"当前阶段：{self.current_phase}\n"
            f"本阶段允许的 action_type：{action_line}\n"
            f"当前位置：{agent.current_location}\n"
            f"可用地点：{'、'.join(self.locations) if self.locations else self.scene}\n"
            f"角色状态：active={agent.active}，alive={agent.alive}\n"
            f"必须遵守的公共规则：\n{rules or '- 依据现实常识与场景设定'}\n"
            f"你有权看到的状态：\n{chr(10).join(visible_variables) or '- 暂无'}\n"
            f"结束条件：\n{conditions or '- 尚未声明'}"
        )

    def get_recent_history(self, k: int = 15, public_only: bool = False) -> str:
        history = (
            [msg for msg in self.history if msg.visibility == "public"]
            if public_only
            else self.history
        )
        recent = history[-k:]
        return "\n".join(msg.as_observation() for msg in recent) or "尚无对话。"

    def public_profiles_for(self, active_agent: str) -> str:
        profiles = [
            f"### {agent.name}\n{agent.public_profile}"
            for agent in self.agents.values()
            if agent.name != active_agent
        ]
        return "\n\n".join(profiles) or "当前场景没有其他角色。"
