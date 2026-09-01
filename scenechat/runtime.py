from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .models import AgentState, MEMORY_TYPES, SimulationState


SAFE_FREE_ACTIONS = {"speak", "act", "observe", "pass"}
TARGETED_ACTIONS = {"vote", "eliminate", "inspect", "protect", "heal", "poison"}
CONVERSATION_MOVES = {
    "statement", "answer", "question", "request", "challenge", "deflect",
    "support", "reveal", "acknowledge", "silence",
}


@dataclass
class Intent:
    actor: str
    action_type: str
    action: str
    speech: str = ""
    target: str = ""
    ability: str = ""
    private_reason: str = ""
    expected_effect: str = ""
    proposed_patch: list[dict[str, Any]] = field(default_factory=list)
    relationship_updates: dict[str, str] = field(default_factory=dict)
    addressed_to: list[str] = field(default_factory=list)
    reply_to_event_id: str = ""
    conversation_move: str = "statement"
    urgency: float = 0.0
    short_term_state: dict[str, Any] = field(default_factory=dict)
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, actor: str, data: dict[str, Any]) -> "Intent":
        proposed = data.get("proposed_patch") or data.get("state_patch") or []
        try:
            urgency = max(0.0, min(float(data.get("urgency", 0.0)), 1.0))
        except (TypeError, ValueError):
            urgency = 0.0
        return cls(
            actor=actor,
            action_type=str(data.get("action_type") or "speak").strip(),
            action=str(data.get("action") or "观察局势").strip(),
            speech=str(data.get("speech") or "").strip(),
            target=str(data.get("target") or "").strip(),
            ability=str(data.get("ability") or "").strip(),
            private_reason=str(data.get("private_reason") or "").strip(),
            expected_effect=str(data.get("expected_effect") or "").strip(),
            proposed_patch=[item for item in proposed if isinstance(item, dict)]
            if isinstance(proposed, list)
            else [],
            relationship_updates={
                str(key): str(value)
                for key, value in (data.get("relationship_updates") or {}).items()
            } if isinstance(data.get("relationship_updates"), dict) else {},
            addressed_to=[
                str(item).strip() for item in data.get("addressed_to") or []
                if str(item).strip()
            ] if isinstance(data.get("addressed_to"), list) else [],
            reply_to_event_id=str(data.get("reply_to_event_id") or "").strip(),
            conversation_move=str(data.get("conversation_move") or "statement").strip(),
            urgency=urgency,
            short_term_state=(
                dict(data.get("short_term_state"))
                if isinstance(data.get("short_term_state"), dict)
                else {}
            ),
            memory_candidates=[
                dict(item) for item in data.get("memory_candidates") or []
                if isinstance(item, dict)
            ] if isinstance(data.get("memory_candidates"), list) else [],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StatePatch:
    operations: list[dict[str, Any]] = field(default_factory=list)

    def add(self, op: str, **values: Any) -> None:
        self.operations.append({"op": op, **values})

    def public_updates(self) -> dict[str, str]:
        updates: dict[str, str] = {}
        for item in self.operations:
            op = item.get("op")
            if op in {"set_world", "increment_world"}:
                updates[str(item.get("key") or "状态")] = str(
                    item.get("value", item.get("amount", "已更新"))
                )
            elif op == "move_agent":
                updates[f"{item.get('target')}位置"] = str(item.get("value") or "")
            elif op == "set_agent_status":
                updates[f"{item.get('target')}状态"] = f"{item.get('key')}={item.get('value')}"
            elif op == "record_vote":
                updates[f"{item.get('actor')}投票"] = str(item.get("target") or "")
            elif op == "set_phase":
                updates["phase"] = str(item.get("value") or "")
        return updates


@dataclass
class Resolution:
    accepted: bool
    reason: str
    intent: Intent
    patch: StatePatch = field(default_factory=StatePatch)
    visibility: str = "public"
    visibility_scopes: list[str] = field(default_factory=lambda: ["public"])
    location: str = ""
    participants: list[str] = field(default_factory=list)
    individual_observations: dict[str, str] = field(default_factory=dict)
    narration: str = ""
    end_signal: bool = False
    end_reason: str = ""


class IntentResolver:
    """Validate model intents and produce the only authoritative state patch."""

    def resolve(self, state: SimulationState, intent: Intent) -> Resolution:
        actor = state.agents.get(intent.actor)
        if actor is None:
            return Resolution(False, "行动者不存在", intent)
        if not actor.eligible:
            return Resolution(False, "行动者已经离场或失去行动资格", intent)

        phase = state.phase_specs.get(state.current_phase)
        allowed_actions = list(getattr(phase, "allowed_action_types", []) or [])
        if allowed_actions and intent.action_type not in allowed_actions:
            return Resolution(
                False,
                f"当前阶段“{state.current_phase}”不允许 {intent.action_type}",
                intent,
            )

        ability = actor.ability_by_reference(intent.ability) if intent.ability else None
        if intent.ability and ability is None:
            return Resolution(False, "角色不具备所声明的能力", intent)
        if ability is not None:
            if not ability.available:
                return Resolution(False, f"能力“{ability.name}”已经没有剩余次数", intent)
            if ability.phases and state.current_phase not in ability.phases:
                return Resolution(False, f"能力“{ability.name}”不能在当前阶段使用", intent)
            if ability.action_type not in {"act", intent.action_type}:
                return Resolution(False, f"能力“{ability.name}”与行动类型不匹配", intent)

        rule = self._matching_rule(state, actor, intent)
        if intent.action_type not in SAFE_FREE_ACTIONS and ability is None and rule is None:
            return Resolution(False, "该行动没有对应的角色能力或场景规则", intent)

        target_error = self._validate_target(state, actor, intent, ability, rule)
        if target_error:
            return Resolution(False, target_error, intent)
        self._sanitize_conversation_metadata(state, actor, intent)

        patch = StatePatch()
        observations: dict[str, str] = {}
        if ability is not None:
            patch.add("consume_ability", target=actor.name, key=ability.id, amount=1)
            for effect in ability.effects:
                self._expand_effect(patch, effect, intent)
        if rule is not None:
            for effect in rule.effects:
                self._expand_effect(patch, asdict(effect), intent)

        self._apply_builtin_effects(state, actor, intent, patch, observations)
        self._append_phase_transition(state, actor, intent, patch)
        scopes = list(
            getattr(ability, "visibility", [])
            or getattr(rule, "visibility", [])
            or ["public"]
        )
        return Resolution(
            True,
            "行动已通过规则校验",
            intent,
            patch=patch,
            visibility=scopes[0] if len(scopes) == 1 else "scoped",
            visibility_scopes=scopes,
            location=actor.current_location,
            individual_observations=observations,
        )

    @staticmethod
    def _sanitize_conversation_metadata(
        state: SimulationState,
        actor: AgentState,
        intent: Intent,
    ) -> None:
        if intent.conversation_move not in CONVERSATION_MOVES:
            intent.conversation_move = "statement" if intent.speech else "silence"
        try:
            intent.urgency = max(0.0, min(float(intent.urgency), 1.0))
        except (TypeError, ValueError):
            intent.urgency = 0.0

        valid_addressees = []
        for name in intent.addressed_to:
            target = state.agents.get(name)
            if (
                target is not None
                and target.name != actor.name
                and target.eligible
                and target.current_location == actor.current_location
                and target.name not in valid_addressees
            ):
                valid_addressees.append(target.name)
        if intent.target in state.agents and intent.target != actor.name:
            target = state.agents[intent.target]
            if target.current_location == actor.current_location and target.eligible:
                if target.name not in valid_addressees:
                    valid_addressees.append(target.name)

        visible_recent = {
            message.event_id: message
            for message in state.history[-30:]
            if state._agent_can_observe(actor, message)
        }
        if intent.reply_to_event_id not in visible_recent:
            intent.reply_to_event_id = ""
        if (
            not intent.reply_to_event_id
            and actor.pending_intents
            and intent.conversation_move in {"answer", "deflect", "challenge", "acknowledge"}
        ):
            def pending_priority(item: dict[str, Any]) -> tuple[float, int]:
                try:
                    urgency = float(item.get("urgency", 0) or 0)
                except (TypeError, ValueError):
                    urgency = 0.0
                try:
                    created_at = int(item.get("created_at_turn", 0) or 0)
                except (TypeError, ValueError):
                    created_at = 0
                return urgency, created_at

            pending = sorted(
                actor.pending_intents,
                key=pending_priority,
                reverse=True,
            )[0]
            candidate = str(pending.get("event_id") or "")
            if candidate in visible_recent:
                intent.reply_to_event_id = candidate
        if intent.reply_to_event_id:
            speaker = visible_recent[intent.reply_to_event_id].speaker
            if speaker in state.agents and speaker != actor.name and speaker not in valid_addressees:
                valid_addressees.append(speaker)
        intent.addressed_to = valid_addressees[:6]

        source = intent.short_term_state
        emotion_source = source.get("emotion") if isinstance(source.get("emotion"), dict) else {}
        label = str(emotion_source.get("label") or "").strip()[:80]
        try:
            intensity = max(0.0, min(float(emotion_source.get("intensity", actor.emotion_intensity)), 1.0))
        except (TypeError, ValueError):
            intensity = actor.emotion_intensity

        def text_list(key: str, maximum: int) -> list[str]:
            values = source.get(key)
            if not isinstance(values, list):
                return []
            return [str(item).strip()[:240] for item in values if str(item).strip()][:maximum]

        intent.short_term_state = {
            "emotion": {
                "label": label,
                "intensity": intensity,
                "cause_event_id": intent.reply_to_event_id,
            },
            "conversation_goal": str(source.get("conversation_goal") or "").strip()[:240],
            "commitments_add": text_list("commitments_add", 4),
            "commitments_resolve": text_list("commitments_resolve", 4),
        }

        candidates = []
        for item in intent.memory_candidates[:8]:
            memory_type = str(item.get("type") or "").strip()
            content = str(item.get("content") or "").strip()[:500]
            if memory_type not in MEMORY_TYPES or not content:
                continue
            try:
                importance = max(1, min(int(item.get("importance", 1)), 5))
            except (TypeError, ValueError):
                importance = 1
            related_agents = []
            for name in item.get("related_agents") or []:
                value = str(name).strip()
                if value in state.agents and value != actor.name and value not in related_agents:
                    related_agents.append(value)
            source_event_id = str(item.get("source_event_id") or "").strip()
            if source_event_id not in visible_recent:
                source_event_id = intent.reply_to_event_id
            candidates.append({
                "type": memory_type,
                "content": content,
                "importance": importance,
                "related_agents": related_agents[:6],
                "source_event_id": source_event_id,
            })
        intent.memory_candidates = candidates

        intent.relationship_updates = {
            target: str(update).strip()[:300]
            for target, update in intent.relationship_updates.items()
            if target in state.agents and target != actor.name and str(update).strip()
        }

    def resolve_director_event(
        self,
        state: SimulationState,
        *,
        narration: str,
        visibility: str,
        proposed_updates: dict[str, Any],
        end_signal: bool = False,
        end_reason: str = "",
        location: str = "",
    ) -> Resolution:
        intent = Intent("旁白", "director_event", "场景推进", narration)
        patch = StatePatch()
        if visibility == "public":
            for key, value in list(proposed_updates.items())[:30]:
                if state._valid_world_value(str(key), value):
                    patch.add("set_world", key=str(key), value=value)
            phase = state.phase_specs.get(state.current_phase)
            if phase is not None and (
                getattr(phase, "event_only", False)
                or getattr(phase, "advance_when", "") == "after_event"
            ):
                patch.add("set_phase", value=getattr(phase, "next_phase", ""))
        # Structured rule scenarios end only through deterministic termination
        # rules. Free-form scenes may still use a director-judged natural end.
        has_deterministic_termination = any(
            getattr(rule, "kind", "manual") != "manual"
            for rule in state.termination_rules
        )
        allow_natural_end = bool(
            end_signal and end_reason.strip() and not has_deterministic_termination
        )
        safe_location = location if (not location or not state.locations or location in state.locations) else ""
        return Resolution(
            True,
            "导演事件已校验",
            intent,
            patch=patch,
            visibility=visibility,
            visibility_scopes=[visibility],
            location=safe_location,
            narration=narration,
            end_signal=allow_natural_end,
            end_reason=end_reason.strip() if allow_natural_end else "",
        )

    @staticmethod
    def _matching_rule(state: SimulationState, actor: AgentState, intent: Intent):
        for rule in state.rules:
            if rule.action_type != intent.action_type:
                continue
            if rule.phases and state.current_phase not in rule.phases:
                continue
            if rule.allowed_roles and actor.role not in rule.allowed_roles:
                continue
            return rule
        return None

    @staticmethod
    def _validate_target(state, actor, intent, ability, rule) -> str:
        if intent.action_type == "move":
            if not intent.target:
                return "移动行动需要指定目标地点"
            if state.locations and intent.target not in state.locations:
                return "目标地点不在场景允许地点中"
            return ""
        scope = getattr(ability, "target_scope", "") or getattr(rule, "target_scope", "") or "none"
        if intent.action_type in TARGETED_ACTIONS and not intent.target:
            return "该行动需要指定目标"
        if scope == "none":
            return ""
        if scope == "self":
            return "" if intent.target in {"", actor.name} else "该能力只能以自己为目标"
        target = state.agents.get(intent.target)
        if target is None:
            return "目标角色不存在"
        if not target.eligible and intent.action_type not in {"heal"}:
            return "目标已经离场或不可行动"
        if scope == "same_location" and target.current_location != actor.current_location:
            return "目标不在行动者所在地点"
        return ""

    @staticmethod
    def _expand_effect(patch: StatePatch, effect: dict[str, Any], intent: Intent) -> None:
        op = str(effect.get("op") or "")
        if not op:
            return
        values = {}
        for key in ("key", "value", "target", "amount"):
            value = effect.get(key)
            if value == "$actor":
                value = intent.actor
            elif value == "$target":
                value = intent.target
            elif value == "$value":
                value = intent.expected_effect
            values[key] = value
        patch.add(op, **values)

    @staticmethod
    def _apply_builtin_effects(
        state: SimulationState,
        actor: AgentState,
        intent: Intent,
        patch: StatePatch,
        observations: dict[str, str],
    ) -> None:
        if intent.action_type == "move":
            patch.add("move_agent", target=actor.name, value=intent.target)
        elif intent.action_type == "vote":
            patch.add("record_vote", actor=actor.name, target=intent.target)
            prospective = dict(state.votes)
            prospective[actor.name] = intent.target
            eligible_names = {item.name for item in state.agents.values() if item.eligible}
            if eligible_names and eligible_names.issubset(prospective):
                tally: dict[str, int] = {}
                for target in prospective.values():
                    tally[target] = tally.get(target, 0) + 1
                highest = max(tally.values())
                winners = [name for name, count in tally.items() if count == highest]
                if len(winners) == 1:
                    patch.add("set_agent_status", target=winners[0], key="alive", value=False)
                patch.add("clear_votes")
        elif intent.action_type == "eliminate":
            if intent.target not in state.protected_agents:
                patch.add("set_agent_status", target=intent.target, key="alive", value=False)
            else:
                observations[actor.name] = "你的袭击没有使目标出局。"
        elif intent.action_type == "poison":
            patch.add("set_agent_status", target=intent.target, key="alive", value=False)
        elif intent.action_type == "protect":
            patch.add("protect_agent", target=intent.target)
        elif intent.action_type == "heal":
            patch.add("set_agent_status", target=intent.target, key="alive", value=True)
            patch.add("set_agent_status", target=intent.target, key="active", value=True)
        elif intent.action_type == "inspect":
            target = state.agents[intent.target]
            fact_id = f"inspection:{state.turn_count + 1}:{target.id or target.name}"
            content = f"查验结果：{target.name}的阵营是“{target.faction or target.role}”。"
            patch.add("add_known_fact", target=actor.name, key=fact_id, value=content)
            observations[actor.name] = content

    @staticmethod
    def _append_phase_transition(
        state: SimulationState,
        actor: AgentState,
        intent: Intent,
        patch: StatePatch,
    ) -> None:
        phase = state.phase_specs.get(state.current_phase)
        if phase is None or phase.advance_when == "manual":
            return
        eligible = [
            item.name
            for item in state.agents.values()
            if item.eligible and (not phase.actor_roles or item.role in phase.actor_roles)
        ]
        acted = set(state.phase_action_log)
        acted.add(actor.name)
        should_advance = phase.advance_when == "all_eligible_acted" and set(eligible).issubset(acted)
        if phase.advance_when == "all_active_voted":
            prospective = set(state.votes)
            if intent.action_type == "vote":
                prospective.add(actor.name)
            should_advance = set(eligible).issubset(prospective)
        if should_advance:
            patch.add("set_phase", value=phase.next_phase or "")
