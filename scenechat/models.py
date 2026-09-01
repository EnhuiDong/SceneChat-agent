from dataclasses import dataclass, field
from typing import Any, Dict, List
import uuid

from .visibility import ViewerContext, can_access, normalize_scopes


MAX_AGENT_MEMORY = 80
MAX_AGENT_OBSERVATIONS = 30
MEMORY_SUMMARY_INTERVAL = 8
MEMORY_TYPES = {
    "claim", "clue", "commitment", "relationship_evidence", "revelation", "decision",
}


@dataclass
class Intervention:
    """Persisted director input; applying it remains a separate validated step."""

    raw_text: str
    mode: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    scope: str = "next_scene"
    created_at_turn: int = 0
    effective_after_turn: int = 0
    status: str = "preview"
    normalized_directive: str = ""
    event_narration: str = ""
    visibility: str = "public"
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    proposed_patch: List[Dict[str, Any]] = field(default_factory=list)
    expires_after_turns: int | None = 1
    applied_at_turn: int | None = None

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "Intervention":
        mode = str(data.get("mode") or "guidance")
        if mode not in {"guidance", "event", "override"}:
            mode = "guidance"
        scope = str(data.get("scope") or "next_scene")
        if scope not in {"next_scene", "turns", "persistent"}:
            scope = "next_scene"
        status = str(data.get("status") or "preview")
        if status not in {"preview", "pending", "applied", "cancelled", "rejected"}:
            status = "preview"

        def optional_int(key: str, default: int | None) -> int | None:
            raw_value = data.get(key, default)
            if raw_value in (None, ""):
                return None
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return default

        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            raw_text=str(data.get("raw_text") or "").strip(),
            mode=mode,
            scope=scope,
            created_at_turn=optional_int("created_at_turn", 0) or 0,
            effective_after_turn=optional_int("effective_after_turn", 0) or 0,
            status=status,
            normalized_directive=str(data.get("normalized_directive") or "").strip(),
            event_narration=str(data.get("event_narration") or "").strip(),
            visibility=(
                str(data.get("visibility") or "public")
                if str(data.get("visibility") or "public") in {"public", "audience_only"}
                else "public"
            ),
            conflicts=[
                item for item in data.get("conflicts") or [] if isinstance(item, dict)
            ],
            proposed_patch=[
                item for item in data.get("proposed_patch") or [] if isinstance(item, dict)
            ],
            expires_after_turns=optional_int("expires_after_turns", 1),
            applied_at_turn=optional_int("applied_at_turn", None),
        )


@dataclass
class ArcState:
    """Future-facing pacing state; progress is measured separately from pace."""

    pace: int = 50
    progress: float = 0.0
    tension: float = 0.0
    target_end_turn: int | None = None
    turns_since_progress: int = 0
    active_beat_ids: List[str] = field(default_factory=list)
    resolved_beat_ids: List[str] = field(default_factory=list)
    skipped_beat_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any] | None) -> "ArcState":
        source = data if isinstance(data, dict) else {}
        try:
            pace = max(0, min(int(source.get("pace", 50)), 100))
        except (TypeError, ValueError):
            pace = 50

        def bounded_float(key: str) -> float:
            try:
                return max(0.0, min(float(source.get(key, 0.0)), 1.0))
            except (TypeError, ValueError):
                return 0.0

        raw_target = source.get("target_end_turn")
        try:
            target = int(raw_target) if raw_target not in (None, "") else None
        except (TypeError, ValueError):
            target = None
        try:
            turns_since_progress = max(
                0, int(source.get("turns_since_progress", 0) or 0)
            )
        except (TypeError, ValueError):
            turns_since_progress = 0
        return cls(
            pace=pace,
            progress=bounded_float("progress"),
            tension=bounded_float("tension"),
            target_end_turn=target,
            turns_since_progress=turns_since_progress,
            active_beat_ids=[str(item) for item in source.get("active_beat_ids") or []],
            resolved_beat_ids=[str(item) for item in source.get("resolved_beat_ids") or []],
            skipped_beat_ids=[str(item) for item in source.get("skipped_beat_ids") or []],
        )


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
    memory_type: str = "event"
    related_agents: List[str] = field(default_factory=list)
    phase: str = ""
    active: bool = True


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
    voice_profile: Dict[str, Any] = field(default_factory=dict)
    current_emotion: str = "平静"
    emotion_intensity: float = 0.2
    emotion_cause_event_id: str = ""
    current_conversation_goal: str = ""
    pending_commitments: List[str] = field(default_factory=list)
    unanswered_questions: List[Dict[str, Any]] = field(default_factory=list)
    last_addressed_by: str = ""
    pending_intents: List[Dict[str, Any]] = field(default_factory=list)
    memories: List[MemoryRecord] = field(default_factory=list)
    memory_summaries: List[Dict[str, Any]] = field(default_factory=list)
    initiative: int = 0

    def observe(self, message: Message, phase: str = "") -> None:
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
                memory_type="event",
                related_agents=[message.speaker] if message.speaker != self.name else [],
                phase=phase,
            )
        )
        self._trim_memories()

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
        self._trim_memories()

    def _trim_memories(self) -> None:
        recent_events = [item for item in self.memories if item.memory_type == "event"][-30:]
        structured = [item for item in self.memories if item.memory_type != "event"][-50:]
        self.memories = sorted(
            recent_events + structured,
            key=lambda item: item.created_at_turn,
        )[-MAX_AGENT_MEMORY:]

    def remember_structured(
        self,
        content: str,
        *,
        memory_type: str,
        importance: int,
        event_id: str,
        turn: int,
        phase: str,
        related_agents: List[str] | None = None,
    ) -> None:
        value = content.strip()[:500]
        kind = memory_type if memory_type in MEMORY_TYPES else "decision"
        try:
            priority = max(1, min(int(importance), 5))
        except (TypeError, ValueError):
            priority = 1
        minimum_importance = {
            "claim": 2,
            "clue": 2,
            "commitment": 1,
            "relationship_evidence": 2,
            "revelation": 2,
            "decision": 3,
        }[kind]
        if not value or priority < minimum_importance:
            return
        normalized = "".join(value.split()).lower()
        duplicate = next((
            item for item in reversed(self.memories[-20:])
            if item.memory_type == kind
            and "".join(item.content.split()).lower() == normalized
        ), None)
        if duplicate:
            duplicate.importance = max(duplicate.importance, priority)
            duplicate.active = True
            duplicate.created_at_turn = max(duplicate.created_at_turn, turn)
            return
        self.memories.append(MemoryRecord(
            event_id=event_id or f"memory-{uuid.uuid4().hex}",
            content=value,
            source="memory_candidate",
            visibility=[f"agent:{self.name}"],
            importance=priority,
            created_at_turn=turn,
            memory_type=kind,
            related_agents=list(related_agents or [])[:6],
            phase=phase,
        ))
        self._trim_memories()

    def resolve_structured_memory(self, memory_type: str, content: str) -> None:
        normalized = "".join(content.split()).lower()
        for item in self.memories:
            if item.memory_type == memory_type and "".join(item.content.split()).lower() == normalized:
                item.active = False

    def layered_memory(self, focus_agents: List[str] | None = None, limit: int = 14) -> str:
        focus = set(focus_agents or [])
        summarized_through = max(
            [int(item.get("through_turn", 0) or 0) for item in self.memory_summaries]
            or [0]
        )
        selected = [
            item for item in self.memories
            if item.active and item.memory_type != "event" and item.importance >= 2
            and (
                item.created_at_turn > summarized_through
                or item.memory_type == "commitment"
                or item.importance >= 5
                or bool(focus.intersection(item.related_agents))
            )
        ]
        selected.sort(
            key=lambda item: (
                bool(focus.intersection(item.related_agents)),
                item.importance,
                item.created_at_turn,
            ),
            reverse=True,
        )
        labels = {
            "claim": "他人主张", "clue": "线索", "commitment": "承诺",
            "relationship_evidence": "关系证据", "revelation": "已披露信息",
            "decision": "关键决定",
        }
        lines = [
            f"- [{labels.get(item.memory_type, item.memory_type)}|重要度 {item.importance}] {item.content}"
            for item in selected[:limit]
        ]
        summaries = [str(item.get("content") or "") for item in self.memory_summaries[-2:]]
        parts = summaries + lines
        return "\n".join(part for part in parts if part) or "- 暂无需要长期保留的故事记忆"

    def refresh_memory_summary(self, phase: str, turn: int, *, force: bool = False) -> None:
        previous_turn = int(self.memory_summaries[-1].get("through_turn", 0)) if self.memory_summaries else 0
        previous_phase = str(self.memory_summaries[-1].get("phase") or "") if self.memory_summaries else ""
        if not self.memory_summaries and not force and turn < MEMORY_SUMMARY_INTERVAL:
            return
        if not force and previous_phase == phase and turn - previous_turn < MEMORY_SUMMARY_INTERVAL:
            return
        window = [item for item in self.memories if item.created_at_turn > previous_turn]
        important = [item for item in window if item.active and item.importance >= 2][-8:]
        if important:
            body = "；".join(f"{item.memory_type}:{item.content[:120]}" for item in important)
        else:
            body = "；".join(
                item.content for item in window if item.memory_type == "event"
            )[-1200:]
        if not body:
            return
        self.memory_summaries.append({
            "phase": phase,
            "through_turn": turn,
            "content": f"- 阶段摘要（{phase}，截至第 {turn} 回合）：{body}",
        })
        self.memory_summaries = self.memory_summaries[-6:]

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
        self.revision = 0
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
        self.interventions: List[Intervention] = []
        self.arc_state = ArcState()
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
        phase_before = self.current_phase
        self.history.append(msg)
        self.turn_count += 1
        if msg.kind in {"narration", "intervention"}:
            self.narration_count += 1
        else:
            self.agent_turn_count += 1

        # Observation is recipient-specific. A public local event reaches only
        # people at that location; role/agent scopes and explicit private
        # observations are checked in code rather than entrusted to the model.
        for agent in self.agents.values():
            if self._agent_can_observe(agent, msg):
                agent.observe(msg, phase=self.current_phase)

        active_agent = self.agents.get(msg.speaker)
        if active_agent is not None:
            if msg.memory:
                active_agent.remember_structured(
                    msg.memory,
                    memory_type="decision",
                    importance=3,
                    event_id=msg.event_id,
                    turn=msg.turn,
                    phase=self.current_phase,
                )
            for target, update in msg.relationship_updates.items():
                if target in self.agents and target != active_agent.name and update.strip():
                    active_agent.relationships[target] = update.strip()
                    active_agent.remember_structured(
                        f"关于{target}：{update.strip()}",
                        memory_type="relationship_evidence",
                        importance=3,
                        event_id=msg.event_id,
                        turn=msg.turn,
                        phase=self.current_phase,
                        related_agents=[target],
                    )
            self._update_conversation_state(active_agent, msg)

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

        intervention_ids = []
        if isinstance(msg.intent, dict):
            intervention_ids.extend(msg.intent.get("intervention_ids") or [])
            if msg.intent.get("intervention_id"):
                intervention_ids.append(msg.intent["intervention_id"])
        for intervention in self.interventions:
            if intervention.id in intervention_ids and intervention.status == "pending":
                intervention.status = "applied"
                intervention.applied_at_turn = msg.turn

        self.evaluate_termination()
        # Arc tracking is deterministic and derived from the committed message.
        # Importing lazily avoids coupling the persistence models to orchestration.
        from .pacing import update_arc_after_message

        update_arc_after_message(self, msg)
        phase_changed = self.current_phase != phase_before
        for agent in self.agents.values():
            if self._agent_can_observe(agent, msg):
                agent.refresh_memory_summary(
                    self.current_phase,
                    msg.turn,
                    force=phase_changed,
                )
        self.bump_revision()

    def _update_conversation_state(self, actor: AgentState, msg: Message) -> None:
        """Commit validated dialogue metadata without granting world authority."""

        intent = msg.intent if isinstance(msg.intent, dict) else {}
        reply_to = str(intent.get("reply_to_event_id") or "").strip()
        if reply_to:
            actor.pending_intents = [
                item for item in actor.pending_intents
                if str(item.get("event_id") or "") != reply_to
            ]
            if not actor.pending_intents:
                actor.last_addressed_by = ""
            for agent in self.agents.values():
                if not self._agent_can_observe(agent, msg):
                    continue
                agent.unanswered_questions = [
                    item for item in agent.unanswered_questions
                    if str(item.get("event_id") or "") != reply_to
                ]

        short_term = intent.get("short_term_state")
        if isinstance(short_term, dict):
            emotion = short_term.get("emotion")
            if isinstance(emotion, dict):
                label = str(emotion.get("label") or "").strip()
                if label:
                    actor.current_emotion = label[:80]
                    actor.emotion_cause_event_id = (
                        str(emotion.get("cause_event_id") or reply_to or msg.event_id)[:80]
                    )
                try:
                    actor.emotion_intensity = max(
                        0.0, min(float(emotion.get("intensity", actor.emotion_intensity)), 1.0)
                    )
                except (TypeError, ValueError):
                    pass
            goal = str(short_term.get("conversation_goal") or "").strip()
            if goal:
                actor.current_conversation_goal = goal[:240]
            resolved = {
                str(item).strip() for item in short_term.get("commitments_resolve") or []
                if str(item).strip()
            }
            if resolved:
                actor.pending_commitments = [
                    item for item in actor.pending_commitments if item not in resolved
                ]
                for item in resolved:
                    actor.resolve_structured_memory("commitment", item)
            for value in short_term.get("commitments_add") or []:
                commitment = str(value).strip()[:240]
                if commitment and commitment not in actor.pending_commitments:
                    actor.pending_commitments.append(commitment)
                    actor.remember_structured(
                        commitment,
                        memory_type="commitment",
                        importance=4,
                        event_id=msg.event_id,
                        turn=msg.turn,
                        phase=self.current_phase,
                    )
            actor.pending_commitments = actor.pending_commitments[-12:]

        for candidate in intent.get("memory_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            actor.remember_structured(
                str(candidate.get("content") or ""),
                memory_type=str(candidate.get("type") or "decision"),
                importance=candidate.get("importance", 1),
                event_id=str(candidate.get("source_event_id") or msg.event_id),
                turn=msg.turn,
                phase=self.current_phase,
                related_agents=[
                    str(name) for name in candidate.get("related_agents") or []
                    if str(name) in self.agents and str(name) != actor.name
                ],
            )

        try:
            actor.initiative = max(0, min(round(float(intent.get("urgency", 0.0)) * 100), 100))
        except (TypeError, ValueError):
            actor.initiative = 0

        move = str(intent.get("conversation_move") or "statement")
        addressed_to = [
            str(name) for name in intent.get("addressed_to") or []
            if str(name) in self.agents and str(name) != actor.name
        ]
        summary = (msg.speech or msg.action).strip()[:300]
        if not summary or move not in {"question", "request", "challenge"}:
            return
        obligation = {
            "event_id": msg.event_id,
            "speaker": actor.name,
            "move": move,
            "summary": summary,
            "urgency": max(0.0, min(actor.initiative / 100, 1.0)),
            "created_at_turn": msg.turn,
        }
        delivered_to = []
        for name in addressed_to:
            target = self.agents[name]
            if not self._agent_can_observe(target, msg):
                continue
            delivered_to.append(name)
            target.last_addressed_by = actor.name
            target.pending_intents = [
                item for item in target.pending_intents
                if str(item.get("event_id") or "") != msg.event_id
            ]
            target.pending_intents.append(dict(obligation))
            target.pending_intents = target.pending_intents[-12:]
        if move == "question" and delivered_to:
            actor.unanswered_questions.append({
                "event_id": msg.event_id,
                "addressed_to": delivered_to,
                "question": summary,
                "created_at_turn": msg.turn,
            })
            actor.unanswered_questions = actor.unanswered_questions[-12:]

    def bump_revision(self) -> int:
        self.revision += 1
        return self.revision

    def register_intervention(self, intervention: Intervention) -> None:
        self.interventions.append(intervention)
        self.bump_revision()

    def set_pace(self, value: int) -> None:
        self.arc_state.pace = max(0, min(int(value), 100))
        from .pacing import initialize_arc

        initialize_arc(self, reset_horizon=True)
        self.bump_revision()

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

    def public_world_state(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        public_viewer = ViewerContext()
        for key, value in self.world_state.items():
            field_spec = self.state_schema.get(key)
            scopes = getattr(field_spec, "visibility", ["public"])
            if can_access(scopes, public_viewer):
                result[key] = value
        return result

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
