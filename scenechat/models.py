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
DEFAULT_RELATIONSHIP_DIMENSIONS = {
    "cooperation": {
        "label": "合作倾向", "low_label": "对抗", "high_label": "协作",
        "description": "双方当前更倾向相互阻碍还是共同配合。",
    },
    "confidence": {
        "label": "可靠判断", "low_label": "不可靠", "high_label": "可靠",
        "description": "该角色认为对方的言行与能力有多可靠。",
    },
    "regard": {
        "label": "重视程度", "low_label": "漠视", "high_label": "重视",
        "description": "该角色在决策中有多重视对方及其处境。",
    },
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
    relationship_updates: Dict[str, Any] = field(default_factory=dict)
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
class BeliefRecord:
    """A character-owned proposition; it is not automatically world truth."""

    content: str
    id: str = field(default_factory=lambda: f"belief-{uuid.uuid4().hex}")
    source_agent: str = ""
    source_event_id: str = ""
    epistemic_status: str = "heard"
    confidence: float = 0.5
    related_agents: List[str] = field(default_factory=list)
    created_at_turn: int = 0
    active: bool = True
    supersedes: str = ""


@dataclass
class DialogueObligation:
    """One directed conversational duty inside a topic thread."""

    source_event_id: str
    requester: str
    target: str
    move: str
    summary: str
    id: str = field(default_factory=lambda: f"obligation-{uuid.uuid4().hex}")
    urgency: float = 0.0
    status: str = "open"
    resolution_event_id: str = ""
    created_at_turn: int = 0
    updated_at_turn: int = 0


@dataclass
class ConversationThread:
    """A topic-scoped interaction chain shared by observable participants."""

    topic: str
    id: str = field(default_factory=lambda: f"thread-{uuid.uuid4().hex}")
    status: str = "active"
    participants: List[str] = field(default_factory=list)
    source_event_ids: List[str] = field(default_factory=list)
    obligations: List[DialogueObligation] = field(default_factory=list)
    claim_ids: List[str] = field(default_factory=list)
    tension: float = 0.0
    created_at_turn: int = 0
    last_active_turn: int = 0


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
    relationship_dynamics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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
    core_beliefs: List[str] = field(default_factory=list)
    belief_records: List[BeliefRecord] = field(default_factory=list)
    voice_profile: Dict[str, Any] = field(default_factory=dict)
    current_emotion: str = "平静"
    emotion_intensity: float = 0.2
    emotion_cause_event_id: str = ""
    current_conversation_goal: str = ""
    disclosure_pressure: float = 0.0
    disclosure_pressure_by_thread: Dict[str, float] = field(default_factory=dict)
    active_thread_ids: List[str] = field(default_factory=list)
    pending_commitments: List[str] = field(default_factory=list)
    unanswered_questions: List[Dict[str, Any]] = field(default_factory=list)
    last_addressed_by: str = ""
    pending_intents: List[Dict[str, Any]] = field(default_factory=list)
    conversation_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    memories: List[MemoryRecord] = field(default_factory=list)
    memory_summaries: List[Dict[str, Any]] = field(default_factory=list)
    initiative: int = 0

    def set_disclosure_pressure(self, thread_id: str, value: float) -> float:
        key = str(thread_id or "general")[:120]
        try:
            bounded = max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            bounded = 0.0
        # Refresh insertion order so trimming keeps the most recently touched topics.
        self.disclosure_pressure_by_thread.pop(key, None)
        self.disclosure_pressure_by_thread[key] = round(bounded, 4)
        self.disclosure_pressure_by_thread = dict(
            list(self.disclosure_pressure_by_thread.items())[-12:]
        )
        self.disclosure_pressure = max(
            self.disclosure_pressure_by_thread.values(), default=bounded
        )
        return bounded

    def adjust_disclosure_pressure(self, thread_id: str, delta: float) -> float:
        key = str(thread_id or "general")[:120]
        current = self.disclosure_pressure_by_thread.get(
            key,
            self.disclosure_pressure if not self.disclosure_pressure_by_thread else 0.0,
        )
        try:
            change = max(-0.25, min(float(delta), 0.25))
        except (TypeError, ValueError):
            change = 0.0
        return self.set_disclosure_pressure(key, current + change)

    def remember_belief(
        self,
        content: str,
        *,
        source_agent: str = "",
        source_event_id: str = "",
        epistemic_status: str = "heard",
        confidence: float = 0.5,
        related_agents: List[str] | None = None,
        turn: int = 0,
        supersedes: str = "",
    ) -> BeliefRecord | None:
        value = str(content or "").strip()[:500]
        if not value:
            return None
        allowed = {"heard", "observed", "inferred", "believed", "verified", "disputed", "disproved"}
        status = epistemic_status if epistemic_status in allowed else "heard"
        try:
            certainty = max(0.0, min(float(confidence), 1.0))
        except (TypeError, ValueError):
            certainty = 0.5
        normalized = "".join(value.split()).lower()
        duplicate = next((
            item for item in reversed(self.belief_records[-30:])
            if item.active and "".join(item.content.split()).lower() == normalized
        ), None)
        if duplicate:
            duplicate.confidence = max(duplicate.confidence, certainty)
            status_rank = {
                "heard": 0, "inferred": 1, "believed": 2,
                "observed": 3, "verified": 4,
            }
            if (
                status in {"disputed", "disproved"}
                or (
                    duplicate.epistemic_status in {"disputed", "disproved"}
                    and status == "verified"
                )
                or (
                    duplicate.epistemic_status not in {"disputed", "disproved"}
                    and status_rank.get(status, 0)
                    >= status_rank.get(duplicate.epistemic_status, 0)
                )
            ):
                duplicate.epistemic_status = status
            duplicate.source_event_id = source_event_id or duplicate.source_event_id
            duplicate.created_at_turn = max(duplicate.created_at_turn, turn)
            return duplicate
        if supersedes:
            for item in self.belief_records:
                if item.id == supersedes:
                    item.active = False
        record = BeliefRecord(
            content=value,
            source_agent=str(source_agent)[:120],
            source_event_id=str(source_event_id)[:120],
            epistemic_status=status,
            confidence=certainty,
            related_agents=list(related_agents or [])[:6],
            created_at_turn=max(0, int(turn or 0)),
            supersedes=str(supersedes)[:120],
        )
        self.belief_records.append(record)
        self.belief_records = self.belief_records[-60:]
        return record

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

    def apply_relationship_update(
        self,
        target: str,
        update: Dict[str, Any] | str,
        *,
        event_id: str,
        turn: int,
        phase: str,
        dimension_specs: Dict[str, Dict[str, str]] | None = None,
    ) -> None:
        if isinstance(update, str):
            value = update.strip()[:300]
            if value:
                self.relationships[target] = value
            return
        if not isinstance(update, dict):
            return
        dimensions = dimension_specs or DEFAULT_RELATIONSHIP_DIMENSIONS
        dynamic = self.relationship_dynamics.setdefault(target, {"facets": {}, "evidence": []})
        facets = dynamic.get("facets")
        if not isinstance(facets, dict):
            facets = {}
        # Schema-v2 compatibility: convert the former genre-biased fixed axes
        # into neutral, world-configurable relationship facets.
        def legacy_value(key: str, default: float = 0.5) -> float:
            try:
                return max(0.0, min(float(dynamic.get(key, default)), 1.0))
            except (TypeError, ValueError):
                return default

        legacy_values = {
            "cooperation": 1.0 - legacy_value("suspicion"),
            "confidence": legacy_value("trust"),
            "regard": legacy_value("affinity"),
        }
        for key in dimensions:
            if key not in facets:
                facets[key] = max(0.0, min(legacy_values.get(key, 0.5), 1.0))

        raw_facets = update.get("facets") if isinstance(update.get("facets"), dict) else {}
        if not raw_facets:
            try:
                legacy_suspicion = -float(update.get("suspicion_delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                legacy_suspicion = 0.0
            raw_facets = {
                "confidence": update.get("trust_delta", 0.0),
                "cooperation": legacy_suspicion,
                "regard": update.get("affinity_delta", 0.0),
            }
        deltas: Dict[str, float] = {}
        for key, raw_delta in raw_facets.items():
            if key not in dimensions:
                continue
            try:
                delta = max(-0.2, min(float(raw_delta), 0.2))
                current = max(0.0, min(float(facets.get(key, 0.5)), 1.0))
            except (TypeError, ValueError):
                continue
            facets[key] = round(max(0.0, min(current + delta, 1.0)), 4)
            deltas[key] = delta
        dynamic["facets"] = facets
        # Retain these values for old exports/readers without using them in new prompts.
        dynamic["trust"] = facets.get("confidence", dynamic.get("trust", 0.5))
        dynamic["suspicion"] = 1.0 - facets.get("cooperation", 0.5)
        dynamic["affinity"] = facets.get("regard", dynamic.get("affinity", 0.5))
        note = str(update.get("private_note") or update.get("summary") or "").strip()[:300]
        evidence = dynamic.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        evidence.append({
            "event_id": str(update.get("reason_event_id") or event_id)[:120],
            "turn": turn,
            "note": note,
            "facets": dict(deltas),
            "proposed_facets": dict(update.get("proposed_facets") or raw_facets),
            "cap": update.get("applied_cap", 0.2),
        })
        dynamic["evidence"] = evidence[-12:]
        summary = str(update.get("summary") or "").strip()[:300]
        if summary:
            self.relationships[target] = summary
        evidence_text = note or "；".join(
            f"{dimensions[key].get('label', key)} {value:+.2f}"
            for key, value in deltas.items()
        )
        self.remember_structured(
            f"关于{target}：{evidence_text}",
            memory_type="relationship_evidence",
            importance=3,
            event_id=str(update.get("reason_event_id") or event_id),
            turn=turn,
            phase=phase,
            related_agents=[target],
        )

    def layered_memory(self, focus_agents: List[str] | None = None, limit: int = 14) -> str:
        focus = set(focus_agents or [])
        summary_turns = [0]
        for item in self.memory_summaries:
            if not isinstance(item, dict):
                continue
            try:
                summary_turns.append(max(0, int(item.get("through_turn", 0) or 0)))
            except (TypeError, ValueError):
                continue
        summarized_through = max(summary_turns)
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
        configured_dimensions = {}
        for dimension in list(getattr(world_spec, "relationship_dimensions", []) or []):
            dimension_id = str(getattr(dimension, "id", "") or "").strip()
            if not dimension_id:
                continue
            configured_dimensions[dimension_id] = {
                "label": str(getattr(dimension, "label", dimension_id) or dimension_id),
                "low_label": str(getattr(dimension, "low_label", "低") or "低"),
                "high_label": str(getattr(dimension, "high_label", "高") or "高"),
                "description": str(getattr(dimension, "description", "") or ""),
            }
        self.relationship_dimensions: Dict[str, Dict[str, str]] = (
            configured_dimensions or {
                key: dict(value) for key, value in DEFAULT_RELATIONSHIP_DIMENSIONS.items()
            }
        )
        self.conversation_threads: Dict[str, ConversationThread] = {}
        self.last_scheduler_decision: Dict[str, Any] = {}
        self.phase_action_log: set[str] = set()
        self.votes: Dict[str, str] = {}
        self.pending_events: List[Dict[str, Any]] = []
        self.interventions: List[Intervention] = []
        self.arc_state = ArcState()
        self.protected_agents: set[str] = set()
        self.failed_generation_count = 0
        self.dialogue_quality_retry_count = 0
        self.dialogue_quality_issue_counts: Dict[str, int] = {}
        self.run_status = "running"

        default_location = self.locations[0] if self.locations else scene
        for agent in self.agents.values():
            if not agent.current_location:
                agent.current_location = default_location
            for target, dynamic in list(agent.relationship_dynamics.items()):
                if target not in self.agents or target == agent.name or not isinstance(dynamic, dict):
                    agent.relationship_dynamics.pop(target, None)
                    continue
                facets = dynamic.get("facets") if isinstance(dynamic.get("facets"), dict) else {}
                dynamic["facets"] = {
                    key: max(0.0, min(float(value), 1.0))
                    for key, value in facets.items()
                    if key in self.relationship_dimensions and isinstance(value, (int, float))
                }
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
                if target in self.agents and target != active_agent.name:
                    active_agent.apply_relationship_update(
                        target,
                        update,
                        event_id=msg.event_id,
                        turn=msg.turn,
                        phase=self.current_phase,
                        dimension_specs=self.relationship_dimensions,
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
        expired_question_ids: set[str] = set()
        for agent in self.agents.values():
            fresh_pending = []
            for item in agent.pending_intents:
                try:
                    created_at = int(item.get("created_at_turn", 0) or 0)
                except (AttributeError, TypeError, ValueError):
                    continue
                if created_at >= msg.turn - 12:
                    fresh_pending.append(item)
                    continue
                pair = self.obligation_by_id(str(item.get("obligation_id") or ""))
                if pair is not None and pair[1].status == "open":
                    pair[1].status = "expired"
                    pair[1].updated_at_turn = msg.turn
                    expired_question_ids.add(pair[1].source_event_id)
            agent.pending_intents = fresh_pending[-12:]
            fresh_opportunities = []
            for item in agent.conversation_opportunities:
                try:
                    created_at = int(item.get("created_at_turn", 0) or 0)
                except (AttributeError, TypeError, ValueError):
                    continue
                if created_at >= msg.turn - 4:
                    fresh_opportunities.append(item)
            agent.conversation_opportunities = fresh_opportunities[-8:]
            if self._agent_can_observe(agent, msg):
                agent.refresh_memory_summary(
                    self.current_phase,
                    msg.turn,
                    force=phase_changed,
                )
            for thread_id, pressure in list(agent.disclosure_pressure_by_thread.items()):
                thread = self.conversation_threads.get(thread_id)
                if thread is not None and msg.turn - thread.last_active_turn > 4:
                    agent.set_disclosure_pressure(thread_id, max(0.0, pressure - 0.03))
            agent.active_thread_ids = [
                thread_id for thread_id in agent.active_thread_ids
                if thread_id in self.conversation_threads
                and self.conversation_threads[thread_id].status == "active"
            ][-8:]
        if expired_question_ids:
            for agent in self.agents.values():
                agent.unanswered_questions = [
                    item for item in agent.unanswered_questions
                    if str(item.get("event_id") or "") not in expired_question_ids
                ]
        for thread in self.conversation_threads.values():
            if thread.status == "active" and msg.turn - thread.last_active_turn > 8:
                thread.status = "dormant"
        self.bump_revision()

    def thread_for_event(self, event_id: str) -> ConversationThread | None:
        value = str(event_id or "")
        if not value:
            return None
        return next((
            thread for thread in self.conversation_threads.values()
            if value in thread.source_event_ids
            or any(item.source_event_id == value for item in thread.obligations)
        ), None)

    def obligation_by_id(
        self,
        obligation_id: str,
    ) -> tuple[ConversationThread, DialogueObligation] | None:
        value = str(obligation_id or "")
        for thread in self.conversation_threads.values():
            for obligation in thread.obligations:
                if obligation.id == value:
                    return thread, obligation
        return None

    def active_threads_for(self, agent_name: str) -> List[ConversationThread]:
        threads = [
            thread for thread in self.conversation_threads.values()
            if thread.status == "active" and agent_name in thread.participants
        ]
        return sorted(threads, key=lambda item: item.last_active_turn, reverse=True)

    def _commit_thread_metadata(
        self,
        actor: AgentState,
        msg: Message,
        intent: Dict[str, Any],
        *,
        move: str,
        addressed_to: List[str],
        summary: str,
    ) -> ConversationThread | None:
        thread = self.conversation_threads.get(str(intent.get("thread_id") or ""))
        reply_pair = self.obligation_by_id(str(intent.get("reply_to_obligation_id") or ""))
        if reply_pair is None:
            reply_event = str(intent.get("reply_to_event_id") or "")
            reply_thread = self.thread_for_event(reply_event)
            if reply_thread is not None:
                obligation = next((
                    item for item in reversed(reply_thread.obligations)
                    if item.source_event_id == reply_event
                    and item.target == actor.name
                    and item.status == "open"
                ), None)
                if obligation is not None:
                    reply_pair = (reply_thread, obligation)
        if reply_pair is not None:
            thread, obligation = reply_pair
            requested = str(intent.get("obligation_resolution") or "")
            if requested not in {"responded", "satisfied", "withdrawn"}:
                requested = (
                    "satisfied" if move in {"answer", "reveal"} else "responded"
                )
            obligation.status = requested
            obligation.resolution_event_id = msg.event_id
            obligation.updated_at_turn = msg.turn
            intent["thread_id"] = thread.id
            intent["reply_to_obligation_id"] = obligation.id
            intent["obligation_resolution"] = requested

        if thread is None and move in {"question", "request", "challenge"} and addressed_to:
            thread = ConversationThread(
                id=f"thread-{msg.event_id[:16]}",
                topic=summary[:160] or f"{actor.name}发起的互动",
                participants=[actor.name, *addressed_to],
                source_event_ids=[msg.event_id],
                tension=0.12 if move == "challenge" else 0.05,
                created_at_turn=msg.turn,
                last_active_turn=msg.turn,
            )
            self.conversation_threads[thread.id] = thread
            intent["thread_id"] = thread.id

        if thread is None:
            return None
        thread.status = "active"
        thread.last_active_turn = msg.turn
        if msg.event_id not in thread.source_event_ids:
            thread.source_event_ids.append(msg.event_id)
            thread.source_event_ids = thread.source_event_ids[-40:]
        for name in [actor.name, *addressed_to]:
            if name in self.agents and name not in thread.participants:
                thread.participants.append(name)
        thread.participants = thread.participants[-12:]
        tension_delta = {
            "challenge": 0.12, "question": 0.04, "request": 0.04,
            "deflect": 0.06, "answer": -0.05, "support": -0.06,
            "reveal": -0.08, "acknowledge": -0.03,
        }.get(move, 0.0)
        thread.tension = round(max(0.0, min(thread.tension + tension_delta, 1.0)), 4)
        if thread.id not in actor.active_thread_ids:
            actor.active_thread_ids.append(thread.id)
            actor.active_thread_ids = actor.active_thread_ids[-8:]
        return thread

    def _update_conversation_state(self, actor: AgentState, msg: Message) -> None:
        """Commit validated dialogue metadata without granting world authority."""

        intent = msg.intent if isinstance(msg.intent, dict) else {}
        actor.conversation_opportunities = []
        move = str(intent.get("conversation_move") or "statement")
        addressed_to = [
            str(name) for name in intent.get("addressed_to") or []
            if str(name) in self.agents and str(name) != actor.name
        ]
        summary = (msg.speech or msg.action).strip()[:300]
        thread = self._commit_thread_metadata(
            actor,
            msg,
            intent,
            move=move,
            addressed_to=addressed_to,
            summary=summary,
        )
        thread_id = thread.id if thread is not None else "general"
        reply_to = str(intent.get("reply_to_event_id") or "").strip()
        reply_resolution = str(intent.get("obligation_resolution") or "")
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
                if reply_resolution == "satisfied":
                    remaining_questions = []
                    for item in agent.unanswered_questions:
                        if str(item.get("event_id") or "") != reply_to:
                            remaining_questions.append(item)
                            continue
                        remaining_targets = [
                            name for name in item.get("addressed_to") or []
                            if str(name) != actor.name
                        ]
                        if remaining_targets:
                            item["addressed_to"] = remaining_targets
                            remaining_questions.append(item)
                    agent.unanswered_questions = remaining_questions
                else:
                    for item in agent.unanswered_questions:
                        if str(item.get("event_id") or "") == reply_to:
                            item["status"] = "responded"
                            item["resolution_event_id"] = msg.event_id

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
            try:
                pressure_delta = max(
                    -0.15,
                    min(float(short_term.get("disclosure_pressure_delta", 0.0)), 0.15),
                )
                actor.adjust_disclosure_pressure(
                    str(intent.get("thread_id") or "general"), pressure_delta
                )
            except (TypeError, ValueError):
                pass
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

        for candidate in intent.get("claim_updates") or []:
            if not isinstance(candidate, dict):
                continue
            source_event_id = str(candidate.get("source_event_id") or reply_to or msg.event_id)
            source_message = next((
                item for item in reversed(self.history)
                if item.event_id == source_event_id
            ), None)
            record = actor.remember_belief(
                str(candidate.get("content") or ""),
                source_agent=(
                    str(candidate.get("source_agent") or "")
                    or (source_message.speaker if source_message is not None else actor.name)
                ),
                source_event_id=source_event_id,
                epistemic_status=str(candidate.get("epistemic_status") or "heard"),
                confidence=candidate.get("confidence", 0.5),
                related_agents=[
                    str(name) for name in candidate.get("related_agents") or []
                    if str(name) in self.agents and str(name) != actor.name
                ],
                turn=msg.turn,
                supersedes=str(candidate.get("supersedes") or ""),
            )
            if record is not None and thread is not None and record.id not in thread.claim_ids:
                thread.claim_ids.append(record.id)
                thread.claim_ids = thread.claim_ids[-30:]

        try:
            actor.initiative = max(0, min(round(float(intent.get("urgency", 0.0)) * 100), 100))
        except (TypeError, ValueError):
            actor.initiative = 0

        mentioned_agents = [
            str(name) for name in intent.get("mentioned_agents") or []
            if str(name) in self.agents
            and str(name) != actor.name
            and str(name) not in addressed_to
        ]
        for name in mentioned_agents:
            target = self.agents[name]
            if not summary or not self._agent_can_observe(target, msg):
                continue
            if thread is not None and name not in thread.participants:
                thread.participants.append(name)
                thread.participants = thread.participants[-12:]
            if thread is not None and thread.id not in target.active_thread_ids:
                target.active_thread_ids.append(thread.id)
                target.active_thread_ids = target.active_thread_ids[-8:]
            target.conversation_opportunities.append({
                "event_id": msg.event_id,
                "speaker": actor.name,
                "move": "mention",
                "summary": summary,
                "urgency": max(0.25, min(actor.initiative / 100, 1.0)),
                "created_at_turn": msg.turn,
                "thread_id": thread.id if thread is not None else "",
            })
            target.conversation_opportunities = target.conversation_opportunities[-8:]

        if move == "reveal":
            actor.adjust_disclosure_pressure(thread_id, -0.25)
        elif move == "deflect":
            actor.adjust_disclosure_pressure(thread_id, 0.08)
        elif move in {"answer", "acknowledge"}:
            actor.adjust_disclosure_pressure(thread_id, -0.08)

        if not summary or move not in {"question", "request", "challenge"}:
            if thread is not None and thread.obligations and all(
                item.status in {"satisfied", "withdrawn"}
                for item in thread.obligations
            ):
                thread.status = "resolved"
            return
        obligation = {
            "event_id": msg.event_id,
            "speaker": actor.name,
            "move": move,
            "summary": summary,
            "urgency": max(0.0, min(actor.initiative / 100, 1.0)),
            "created_at_turn": msg.turn,
            "thread_id": thread.id if thread is not None else "",
        }
        delivered_to = []
        for name in addressed_to:
            target = self.agents[name]
            if not self._agent_can_observe(target, msg):
                continue
            delivered_to.append(name)
            target.last_addressed_by = actor.name
            pressure_gain = (0.04 if move == "question" else 0.08) + obligation["urgency"] * 0.12
            target.adjust_disclosure_pressure(thread_id, pressure_gain)
            if thread is not None:
                if thread.id not in target.active_thread_ids:
                    target.active_thread_ids.append(thread.id)
                    target.active_thread_ids = target.active_thread_ids[-8:]
                structured_obligation = DialogueObligation(
                    id=f"obligation-{msg.event_id[:12]}-{target.id or target.name}",
                    source_event_id=msg.event_id,
                    requester=actor.name,
                    target=target.name,
                    move=move,
                    summary=summary,
                    urgency=obligation["urgency"],
                    created_at_turn=msg.turn,
                    updated_at_turn=msg.turn,
                )
                thread.obligations.append(structured_obligation)
                thread.obligations = thread.obligations[-30:]
                obligation["obligation_id"] = structured_obligation.id
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

    def record_dialogue_quality_issues(self, codes: List[str], *, retried: bool) -> None:
        if retried:
            self.dialogue_quality_retry_count += 1
        for code in codes:
            value = str(code).strip()
            if value:
                self.dialogue_quality_issue_counts[value] = (
                    self.dialogue_quality_issue_counts.get(value, 0) + 1
                )

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
