from __future__ import annotations

import json
import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .visibility import normalize_scopes


VALID_VISIBILITIES = {"public", "director_only", "audience_only"}
VALID_SCHEDULERS = {
    "round_robin",
    "phase_order",
    "initiative",
    "urgency_director",
    "event_first",
}
VALID_PATCH_OPERATIONS = {
    "set_world",
    "increment_world",
    "move_agent",
    "set_agent_status",
    "set_resource",
    "consume_resource",
    "set_goal_status",
    "set_relationship",
    "record_vote",
    "clear_votes",
    "set_phase",
    "add_known_fact",
    "consume_ability",
    "protect_agent",
    "clear_protections",
}

ACTION_EFFECT_ALLOWLIST = {
    "move": {"move_agent"},
    "vote": {"record_vote", "clear_votes"},
    "inspect": {"add_known_fact"},
    "protect": {"protect_agent", "clear_protections"},
    "eliminate": {"set_agent_status"},
    "poison": {"set_agent_status"},
    "heal": {"set_agent_status"},
}
COMMON_EFFECT_OPERATIONS = {
    "set_world",
    "increment_world",
    "set_resource",
    "consume_resource",
    "set_goal_status",
    "set_relationship",
}


def _text(value: Any, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return [_text(value)] if _text(value) else []
    return [_text(item) for item in value if _text(item)]


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {_text(key): _text(item) for key, item in value.items() if _text(key)}


def _state_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        _text(key): item
        for key, item in value.items()
        if _text(key) and isinstance(item, (str, int, float, bool))
    }


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response.

    Providers occasionally wrap otherwise valid JSON in a Markdown fence or add
    a short preamble. The decoder keeps the structured pipeline tolerant without
    accepting arbitrary Markdown as a successful structured result.
    """

    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("模型响应中没有 JSON 对象")
    candidate = text[start:]
    try:
        payload, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as original_error:
        # Common provider drift: JavaScript comments, trailing commas, or
        # unquoted simple property names. This is deliberately conservative;
        # semantic repair remains the model retry's responsibility.
        repaired = re.sub(r"/\*.*?\*/|//[^\r\n]*", "", candidate, flags=re.DOTALL)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(
            r'([,{]\s*)([A-Za-z_\u4e00-\u9fff][\w\-\u4e00-\u9fff]*)(\s*:)',
            r'\1"\2"\3',
            repaired,
        )
        try:
            payload, _ = json.JSONDecoder().raw_decode(repaired)
        except json.JSONDecodeError:
            python_literal = re.sub(r"\btrue\b", "True", repaired)
            python_literal = re.sub(r"\bfalse\b", "False", python_literal)
            python_literal = re.sub(r"\bnull\b", "None", python_literal)
            try:
                payload = ast.literal_eval(python_literal)
            except (SyntaxError, ValueError) as exc:
                raise original_error from exc
    if not isinstance(payload, dict):
        raise ValueError("模型响应顶层必须是 JSON 对象")
    return payload


@dataclass
class ConstraintItem:
    id: str
    category: str
    content: str
    source_excerpt: str = ""
    visibility: str = "public"
    locked: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any], index: int) -> "ConstraintItem":
        visibility = _text(data.get("visibility"), "public")
        if visibility not in VALID_VISIBILITIES and not visibility.startswith(
            ("agent:", "role:", "location:")
        ):
            visibility = "director_only"
        return cls(
            id=_text(data.get("id"), f"constraint-{index}"),
            category=_text(data.get("category"), "other"),
            content=_text(data.get("content")),
            source_excerpt=_text(data.get("source_excerpt")),
            visibility=visibility,
            locked=bool(data.get("locked", True)),
        )


@dataclass
class ScenarioBrief:
    input_mode: str
    genre: str
    premise: str
    requested_character_count: int | None = None
    requested_opening_scene: str = ""
    constraints: list[ConstraintItem] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    fixed_canon: list[str] = field(default_factory=list)
    target_beats: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ScenarioBrief":
        raw_count = data.get("requested_character_count")
        try:
            count = int(raw_count) if raw_count not in (None, "") else None
        except (TypeError, ValueError):
            count = None
        constraints = [
            ConstraintItem.from_mapping(item, index)
            for index, item in enumerate(data.get("constraints") or [], start=1)
            if isinstance(item, dict)
        ]
        return cls(
            input_mode=_text(data.get("input_mode"), "partial"),
            genre=_text(data.get("genre"), "互动场景"),
            premise=_text(data.get("premise")),
            requested_character_count=count,
            requested_opening_scene=_text(data.get("requested_opening_scene")),
            constraints=constraints,
            assumptions=_string_list(data.get("assumptions")),
            contradictions=_string_list(data.get("contradictions")),
            fixed_canon=_string_list(data.get("fixed_canon")),
            target_beats=_string_list(data.get("target_beats")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public_dict(self) -> dict[str, Any]:
        return {
            "input_mode": self.input_mode,
            "genre": self.genre,
            "premise": self.premise,
            "requested_character_count": self.requested_character_count,
            "requested_opening_scene": self.requested_opening_scene,
            "constraints": [
                asdict(item) for item in self.constraints if item.visibility == "public"
            ],
        }


@dataclass
class FactSpec:
    id: str
    content: str
    visibility: list[str] = field(default_factory=lambda: ["public"])
    source: str = "world"
    covered_constraint_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], index: int) -> "FactSpec":
        return cls(
            id=_text(data.get("id"), f"fact-{index}"),
            content=_text(data.get("content")),
            visibility=normalize_scopes(data.get("visibility")),
            source=_text(data.get("source"), "world"),
            covered_constraint_ids=_string_list(data.get("covered_constraint_ids")),
        )


@dataclass
class StateFieldSpec:
    key: str
    value_type: str = "string"
    visibility: list[str] = field(default_factory=lambda: ["public"])
    mutable_by: list[str] = field(default_factory=lambda: ["resolver"])
    allowed_values: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, key: str, data: Any) -> "StateFieldSpec":
        if not isinstance(data, dict):
            data = {"value_type": "string"}
        return cls(
            key=key,
            value_type=_text(data.get("value_type"), "string"),
            visibility=normalize_scopes(data.get("visibility")),
            mutable_by=_string_list(data.get("mutable_by")) or ["resolver"],
            allowed_values=_string_list(data.get("allowed_values")),
        )


@dataclass
class PhaseSpec:
    name: str
    scheduler: str = "round_robin"
    allowed_action_types: list[str] = field(default_factory=list)
    actor_roles: list[str] = field(default_factory=list)
    advance_when: str = "all_eligible_acted"
    next_phase: str = ""
    event_only: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any], index: int) -> "PhaseSpec":
        name = _text(data.get("name"), f"阶段 {index}")
        scheduler = _text(data.get("scheduler"), "round_robin")
        if scheduler not in VALID_SCHEDULERS:
            scheduler = "round_robin"
        event_only = bool(data.get("event_only", False))
        if not event_only and scheduler == "event_first" and not data.get("allowed_action_types"):
            event_only = True
        event_name_markers = ("announcement", "resolution", "公布", "结算", "广播", "揭晓")
        if not data.get("allowed_action_types") and any(
            marker in name.lower() for marker in event_name_markers
        ):
            event_only = True
            scheduler = "event_first"
        return cls(
            name=name,
            scheduler=scheduler,
            allowed_action_types=_string_list(data.get("allowed_action_types")),
            actor_roles=_string_list(data.get("actor_roles")),
            advance_when=_text(
                data.get("advance_when"),
                "after_event" if event_only else "all_eligible_acted",
            ),
            next_phase=_text(data.get("next_phase")),
            event_only=event_only,
        )


@dataclass
class PatchTemplate:
    op: str
    key: str = ""
    value: Any = None
    target: str = ""
    amount: int = 1

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PatchTemplate":
        try:
            amount = int(data.get("amount", 1))
        except (TypeError, ValueError):
            amount = 1
        return cls(
            op=_text(data.get("op")),
            key=_text(data.get("key")),
            value=data.get("value"),
            target=_text(data.get("target")),
            amount=amount,
        )


@dataclass
class RuleSpec:
    id: str
    description: str
    action_type: str
    phases: list[str] = field(default_factory=list)
    allowed_roles: list[str] = field(default_factory=list)
    target_scope: str = "same_location"
    effects: list[PatchTemplate] = field(default_factory=list)
    visibility: list[str] = field(default_factory=lambda: ["public"])

    @classmethod
    def from_mapping(cls, data: dict[str, Any], index: int) -> "RuleSpec":
        return cls(
            id=_text(data.get("id"), f"rule-{index}"),
            description=_text(data.get("description")),
            action_type=_text(data.get("action_type"), "act"),
            phases=_string_list(data.get("phases")),
            allowed_roles=_string_list(data.get("allowed_roles")),
            target_scope=_text(data.get("target_scope"), "same_location"),
            effects=[
                PatchTemplate.from_mapping(item)
                for item in data.get("effects") or []
                if isinstance(item, dict)
            ],
            visibility=normalize_scopes(data.get("visibility")),
        )


@dataclass
class TerminationRule:
    id: str
    kind: str
    description: str
    faction: str = ""
    opposing_factions: list[str] = field(default_factory=list)
    key: str = ""
    value: Any = None
    winner: str = ""
    conditions: list["TerminationRule"] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    location: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any], index: int) -> "TerminationRule":
        return cls(
            id=_text(data.get("id"), f"termination-{index}"),
            kind=_text(data.get("kind"), "manual"),
            description=_text(data.get("description")),
            faction=_text(data.get("faction")),
            opposing_factions=_string_list(data.get("opposing_factions")),
            key=_text(data.get("key")),
            value=data.get("value"),
            winner=_text(data.get("winner")),
            conditions=[
                cls.from_mapping(item, child_index)
                for child_index, item in enumerate(data.get("conditions") or [], start=1)
                if isinstance(item, dict)
            ],
            phases=_string_list(data.get("phases")),
            location=_text(data.get("location")),
        )


@dataclass
class WorldSpec:
    title: str
    opening_scene: str
    public_world_markdown: str
    director_notes_markdown: str = ""
    public_rules: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    termination_conditions: list[str] = field(default_factory=list)
    fixed_canon: list[str] = field(default_factory=list)
    target_beats: list[str] = field(default_factory=list)
    covered_constraint_ids: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    facts: list[FactSpec] = field(default_factory=list)
    state_schema: dict[str, StateFieldSpec] = field(default_factory=dict)
    scheduler: str = "round_robin"
    phase_specs: list[PhaseSpec] = field(default_factory=list)
    rules: list[RuleSpec] = field(default_factory=list)
    termination_rules: list[TerminationRule] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "WorldSpec":
        scheduler = _text(data.get("scheduler"), "round_robin")
        if scheduler not in VALID_SCHEDULERS:
            scheduler = "round_robin"
        phases = _string_list(data.get("phases"))
        phase_specs = [
            PhaseSpec.from_mapping(item, index)
            for index, item in enumerate(data.get("phase_specs") or [], start=1)
            if isinstance(item, dict)
        ]
        if not phase_specs:
            phase_specs = [
                PhaseSpec(
                    name=name,
                    scheduler=scheduler,
                    next_phase=phases[(index + 1) % len(phases)] if len(phases) > 1 else name,
                )
                for index, name in enumerate(phases)
            ]
        if not phases:
            phases = [phase.name for phase in phase_specs]
        state_schema = {
            _text(key): StateFieldSpec.from_mapping(_text(key), value)
            for key, value in (data.get("state_schema") or {}).items()
            if _text(key)
        } if isinstance(data.get("state_schema"), dict) else {}
        return cls(
            title=_text(data.get("title"), "未命名互动场景"),
            opening_scene=_text(data.get("opening_scene")),
            public_world_markdown=_text(data.get("public_world_markdown")),
            director_notes_markdown=_text(data.get("director_notes_markdown")),
            public_rules=_string_list(data.get("public_rules")),
            phases=phases,
            initial_state=_state_dict(data.get("initial_state")),
            termination_conditions=_string_list(data.get("termination_conditions")),
            fixed_canon=_string_list(data.get("fixed_canon")),
            target_beats=_string_list(data.get("target_beats")),
            covered_constraint_ids=_string_list(data.get("covered_constraint_ids")),
            locations=_string_list(data.get("locations")),
            facts=[
                FactSpec.from_mapping(item, index)
                for index, item in enumerate(data.get("facts") or [], start=1)
                if isinstance(item, dict)
            ],
            state_schema=state_schema,
            scheduler=scheduler,
            phase_specs=phase_specs,
            rules=[
                RuleSpec.from_mapping(item, index)
                for index, item in enumerate(data.get("rules") or [], start=1)
                if isinstance(item, dict)
            ],
            termination_rules=[
                TerminationRule.from_mapping(item, index)
                for index, item in enumerate(data.get("termination_rules") or [], start=1)
                if isinstance(item, dict)
            ],
        )

    def to_markdown(self, include_director: bool = True) -> str:
        public = self.public_world_markdown.strip()
        if not public.startswith("#"):
            public = f"# {self.title}\n\n{public}".strip()
        state_lines = "\n".join(f"- **{key}**：{value}" for key, value in self.initial_state.items())
        runtime = [
            "## 运行时场景信息",
            f"- **建议开场**：{self.opening_scene}",
        ]
        if self.phases:
            runtime.append(f"- **阶段顺序**：{' → '.join(self.phases)}")
        runtime.append(f"- **调度策略**：{self.scheduler}")
        if state_lines:
            runtime.extend(["", "### 初始公共状态", state_lines])
        result = f"{public}\n\n" + "\n".join(runtime)
        if include_director and self.director_notes_markdown:
            result += (
                "\n\n<!-- SCENECHAT_DIRECTOR_ONLY_BEGIN -->\n"
                "## 导演私密信息（不得提供给无权限角色）\n\n"
                f"{self.director_notes_markdown.strip()}\n"
                "<!-- SCENECHAT_DIRECTOR_ONLY_END -->"
            )
        return result.strip()


@dataclass
class AbilitySpec:
    id: str
    name: str
    action_type: str
    description: str = ""
    phases: list[str] = field(default_factory=list)
    uses: int | None = None
    target_scope: str = "same_location"
    effects: list[PatchTemplate] = field(default_factory=list)
    visibility: list[str] = field(default_factory=lambda: ["public"])

    @classmethod
    def from_value(cls, value: Any, index: int) -> "AbilitySpec":
        if isinstance(value, str):
            return cls(
                id=f"ability-{index}",
                name=value,
                action_type="act",
                description=value,
            )
        data = value if isinstance(value, dict) else {}
        raw_uses = data.get("uses")
        try:
            uses = int(raw_uses) if raw_uses not in (None, "", "unlimited") else None
        except (TypeError, ValueError):
            uses = None
        action_type = _text(data.get("action_type"), "act")
        default_visibility = (
            ["director_only"]
            if action_type in {"inspect", "protect", "eliminate", "poison"}
            else ["public"]
        )
        return cls(
            id=_text(data.get("id"), f"ability-{index}"),
            name=_text(data.get("name"), _text(data.get("description"), f"能力 {index}")),
            action_type=action_type,
            description=_text(data.get("description")),
            phases=_string_list(data.get("phases")),
            uses=uses,
            target_scope=_text(data.get("target_scope"), "same_location"),
            effects=[
                PatchTemplate.from_mapping(item)
                for item in data.get("effects") or []
                if isinstance(item, dict)
            ],
            visibility=normalize_scopes(data.get("visibility") or default_visibility),
        )


def _abilities_from_scalar(value: Any) -> list[AbilitySpec]:
    return [
        AbilitySpec.from_value(item, index)
        for index, item in enumerate(_string_list(value), start=1)
    ]


@dataclass
class CharacterSpec:
    id: str
    name: str
    role: str
    public_identity: str
    public_traits: str
    public_background: str
    private_identity: str
    personality: str
    goals: list[str]
    knowledge: dict[str, str]
    decision_logic: str
    relationships: dict[str, str]
    observation_value: str
    abilities: list[AbilitySpec] = field(default_factory=list)
    covered_constraint_ids: list[str] = field(default_factory=list)
    faction: str = ""
    initial_location: str = ""
    resources: dict[str, Any] = field(default_factory=dict)
    known_fact_ids: list[str] = field(default_factory=list)
    false_beliefs: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], index: int) -> "CharacterSpec":
        return cls(
            id=_text(data.get("id"), f"character-{index}"),
            name=_text(data.get("name")),
            role=_text(data.get("role")),
            public_identity=_text(data.get("public_identity")),
            public_traits=_text(data.get("public_traits")),
            public_background=_text(data.get("public_background")),
            private_identity=_text(data.get("private_identity"), "无额外秘密"),
            personality=_text(data.get("personality")),
            goals=_string_list(data.get("goals")),
            knowledge=_string_dict(data.get("knowledge")),
            decision_logic=_text(data.get("decision_logic")),
            relationships=_string_dict(data.get("relationships")),
            observation_value=_text(data.get("observation_value")),
            abilities=[
                AbilitySpec.from_value(item, ability_index)
                for ability_index, item in enumerate(data.get("abilities") or [], start=1)
            ] if isinstance(data.get("abilities"), list) else _abilities_from_scalar(data.get("abilities")),
            covered_constraint_ids=_string_list(data.get("covered_constraint_ids")),
            faction=_text(data.get("faction")),
            initial_location=_text(data.get("initial_location")),
            resources=dict(data.get("resources") or {}) if isinstance(data.get("resources"), dict) else {},
            known_fact_ids=_string_list(data.get("known_fact_ids")),
            false_beliefs=_string_list(data.get("false_beliefs")),
        )

    @property
    def public_profile(self) -> str:
        return "\n".join(
            [
                f"角色姓名：{self.name}",
                f"公开身份：{self.public_identity}",
                f"公开形象与行为特征：{self.public_traits}",
                f"公开背景：{self.public_background}",
            ]
        )

    def to_markdown(self, index: int) -> str:
        knowledge = "\n".join(f"- **{key}**：{value}" for key, value in self.knowledge.items())
        relationships = "\n".join(
            f"- **{key}**：{value}" for key, value in self.relationships.items()
        )
        goals = "\n".join(f"- {goal}" for goal in self.goals) or "- 依据人设与当前局势行动"
        abilities = "\n".join(
            f"- {ability.name}（{ability.action_type}；"
            f"{'不限次数' if ability.uses is None else f'剩余 {ability.uses} 次'}）"
            for ability in self.abilities
        )
        private_identity = self.private_identity
        if abilities:
            private_identity = f"{private_identity}\n\n**能力与资源**\n{abilities}"
        return f"""## 角色 {index}
### 1. 角色姓名
{self.name}

### 2. 公开身份
{self.public_identity}

### 3. 公开形象与行为特征
{self.public_traits}

### 4. 公开背景
{self.public_background}

### 5. 私密身份与秘密
{private_identity}

### 6. 性格与形成原因
{self.personality}

### 7. 核心目标与顾虑
{goals}

### 8. 初始知识与信息边界
{knowledge or '未提供额外私有知识。'}

### 9. 决策与行为逻辑
{self.decision_logic}

### 10. 对其他角色的初始认知
{relationships or '尚未形成明确的个人认知。'}

### 11. 实验观察价值
{self.observation_value}""".strip()

    def to_public_markdown(self, index: int) -> str:
        return f"""## 角色 {index}
### 角色姓名
{self.name}

### 公开身份
{self.public_identity}

### 公开形象与行为特征
{self.public_traits}

### 公开背景
{self.public_background}""".strip()


@dataclass
class ScenarioPackage:
    brief: ScenarioBrief
    world: WorldSpec
    characters: list[CharacterSpec]
    warnings: list[str] = field(default_factory=list)

    @property
    def worldview_markdown(self) -> str:
        return self.world.to_markdown(include_director=True)

    @property
    def public_worldview_markdown(self) -> str:
        return self.world.to_markdown(include_director=False)

    @property
    def characters_markdown(self) -> str:
        dossiers = [character.to_markdown(index) for index, character in enumerate(self.characters, 1)]
        return "# 互动社会实验角色档案\n\n" + "\n\n".join(dossiers)

    @property
    def public_characters_markdown(self) -> str:
        dossiers = [
            character.to_public_markdown(index)
            for index, character in enumerate(self.characters, 1)
        ]
        return "# 公开角色资料\n\n" + "\n\n".join(dossiers)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScenarioValidationError(ValueError):
    def __init__(self, issues: Iterable[str]):
        self.issues = [issue for issue in issues if issue]
        super().__init__("；".join(self.issues))


def _phase_reference_key(value: str) -> str:
    text = _text(value).lower()
    text = re.sub(
        r"^(?:第?[一二三四五六七八九十百\d]+(?:轮|阶段|幕|回合)?|phase\s*\d+|round\s*\d+)\s*[:：.、\-—_]*",
        "",
        text,
    )
    return re.sub(r"[\s:：.、,，;；\-—_（）()\[\]]+", "", text)


def normalize_scenario_phase_references(package: ScenarioPackage) -> None:
    """Resolve harmless display-name drift to the exact canonical phase names.

    Models sometimes emit ``第一阶段：公开报价`` in one structure and
    ``公开报价`` in another. The mapping only applies when one canonical phase
    is an unambiguous normalized match; genuinely unknown references remain for
    validation to reject.
    """

    phases = list(package.world.phases)
    if not phases:
        return
    keyed = {phase: _phase_reference_key(phase) for phase in phases}

    def resolve(value: str) -> str:
        if value in keyed:
            return value
        key = _phase_reference_key(value)
        if not key:
            return value
        candidates = [
            phase
            for phase, phase_key in keyed.items()
            if key == phase_key
            or (len(key) >= 3 and key in phase_key)
            or (len(phase_key) >= 3 and phase_key in key)
        ]
        return candidates[0] if len(candidates) == 1 else value

    for phase_spec in package.world.phase_specs:
        phase_spec.name = resolve(phase_spec.name)
        phase_spec.next_phase = resolve(phase_spec.next_phase)
    for rule in package.world.rules:
        rule.phases = [resolve(phase) for phase in rule.phases]

    def normalize_termination(rule: TerminationRule) -> None:
        rule.phases = [resolve(phase) for phase in rule.phases]
        for child in rule.conditions:
            normalize_termination(child)

    for termination_rule in package.world.termination_rules:
        normalize_termination(termination_rule)
    for character in package.characters:
        for ability in character.abilities:
            ability.phases = [resolve(phase) for phase in ability.phases]


def validate_scenario_package(package: ScenarioPackage) -> list[str]:
    issues: list[str] = []
    if not package.world.opening_scene:
        issues.append("缺少可直接运行的 opening_scene")
    if not package.world.public_world_markdown:
        issues.append("公共世界设定为空")
    if not package.world.termination_conditions:
        issues.append("缺少场景结束条件")
    if not package.characters:
        issues.append("没有生成任何角色")

    requested_count = package.brief.requested_character_count
    if requested_count is not None and len(package.characters) != requested_count:
        issues.append(f"用户要求 {requested_count} 名角色，实际生成 {len(package.characters)} 名")

    names = [character.name for character in package.characters]
    if any(not name for name in names):
        issues.append("存在缺少姓名的角色")
    if len(set(names)) != len(names):
        issues.append("角色姓名不唯一")

    for character in package.characters:
        if not character.public_identity:
            issues.append(f"角色“{character.name or character.id}”缺少公开身份")
        if not character.decision_logic:
            issues.append(f"角色“{character.name or character.id}”缺少决策逻辑")
        if not character.goals:
            issues.append(f"角色“{character.name or character.id}”缺少目标")
        if (
            character.initial_location
            and package.world.locations
            and character.initial_location not in package.world.locations
        ):
            issues.append(
                f"角色“{character.name or character.id}”的初始地点不在 locations 中"
            )
        ability_ids = [ability.id for ability in character.abilities]
        if len(set(ability_ids)) != len(ability_ids):
            issues.append(f"角色“{character.name or character.id}”存在重复能力 ID")
        for ability in character.abilities:
            unknown_phases = set(ability.phases) - set(package.world.phases)
            if unknown_phases:
                issues.append(
                    f"角色“{character.name}”的能力“{ability.name}”引用未知阶段："
                    f"{', '.join(sorted(unknown_phases))}"
                )
            invalid_ops = [effect.op for effect in ability.effects if effect.op not in VALID_PATCH_OPERATIONS]
            if invalid_ops:
                issues.append(
                    f"角色“{character.name}”的能力“{ability.name}”包含非法状态操作"
                )
            allowed_effects = COMMON_EFFECT_OPERATIONS | ACTION_EFFECT_ALLOWLIST.get(
                ability.action_type,
                set(),
            )
            if any(effect.op not in allowed_effects for effect in ability.effects):
                issues.append(
                    f"角色“{character.name}”的能力“{ability.name}”包含与行动类型不匹配的 effect"
                )

    phase_names = [phase.name for phase in package.world.phase_specs]
    if len(set(phase_names)) != len(phase_names):
        issues.append("阶段名称不唯一")
    rule_ids = [rule.id for rule in package.world.rules]
    if len(set(rule_ids)) != len(rule_ids):
        issues.append("规则 ID 不唯一")
    for rule in package.world.rules:
        if set(rule.phases) - set(package.world.phases):
            issues.append(f"规则“{rule.id}”引用了未知阶段")
        if any(effect.op not in VALID_PATCH_OPERATIONS for effect in rule.effects):
            issues.append(f"规则“{rule.id}”包含非法状态操作")
        allowed_effects = COMMON_EFFECT_OPERATIONS | ACTION_EFFECT_ALLOWLIST.get(
            rule.action_type,
            set(),
        ) | {"set_phase"}
        if any(effect.op not in allowed_effects for effect in rule.effects):
            issues.append(f"规则“{rule.id}”包含与行动类型不匹配的 effect")

    fact_ids = [fact.id for fact in package.world.facts]
    if len(set(fact_ids)) != len(fact_ids):
        issues.append("世界事实 ID 不唯一")

    valid_termination_kinds = {
        "faction_eliminated",
        "faction_parity",
        "world_equals",
        "all_goals_completed",
        "all_active_at_location",
        "all_of",
        "any_of",
        "manual",
    }

    def check_termination(rule: TerminationRule) -> None:
        if rule.kind not in valid_termination_kinds:
            issues.append(f"结束规则“{rule.id}”使用未知 kind")
        if rule.kind in {"all_of", "any_of"} and not rule.conditions:
            issues.append(f"组合结束规则“{rule.id}”缺少 conditions")
        if rule.kind == "world_equals" and rule.key not in package.world.initial_state:
            issues.append(f"结束规则“{rule.id}”引用未知世界状态“{rule.key}”")
        if (
            rule.kind == "all_active_at_location"
            and package.world.locations
            and rule.location not in package.world.locations
        ):
            issues.append(f"结束规则“{rule.id}”引用未知地点")
        if set(rule.phases) - set(package.world.phases):
            issues.append(f"结束规则“{rule.id}”引用未知阶段")
        for child in rule.conditions:
            check_termination(child)

    for termination_rule in package.world.termination_rules:
        check_termination(termination_rule)

    structured_runtime = (
        len(package.world.phases) > 1
        or package.world.scheduler != "round_robin"
        or any(
            ability.action_type != "act"
            for character in package.characters
            for ability in character.abilities
        )
    )
    if structured_runtime:
        if not package.world.phase_specs:
            issues.append("规则型场景缺少 phase_specs")
        if not package.world.rules:
            issues.append("规则型场景缺少可执行 rules")
        if not package.world.termination_rules:
            issues.append("规则型场景缺少结构化 termination_rules")
        for phase in package.world.phase_specs:
            if not phase.allowed_action_types and not phase.event_only:
                issues.append(f"规则型阶段“{phase.name}”缺少 allowed_action_types")
            if (
                not phase.event_only
                and phase.allowed_action_types
                and not set(phase.allowed_action_types).intersection(
                    {"pass", "observe", "speak", "act"}
                )
            ):
                issues.append(
                    f"规则型阶段“{phase.name}”缺少安全兜底行动 pass/observe/speak/act"
                )
            if phase.next_phase and phase.next_phase not in package.world.phases:
                issues.append(f"阶段“{phase.name}”引用未知 next_phase")
        compound_text = " ".join(package.world.termination_conditions)
        if (
            len(package.world.termination_rules) > 1
            and any(marker in compound_text for marker in ("且", "同时", "全部满足", "都完成"))
            and not any(
                rule.kind == "all_of" for rule in package.world.termination_rules
            )
        ):
            issues.append("复合结束条件需要使用 all_of，而不是多个并列顶层规则")

    public_text = "\n".join(
        [package.world.public_world_markdown]
        + [character.public_profile for character in package.characters]
    )
    for constraint in package.brief.constraints:
        if constraint.visibility == "public":
            continue
        for private_text in (constraint.content, constraint.source_excerpt):
            normalized = "".join(private_text.split())
            if len(normalized) >= 8 and normalized in "".join(public_text.split()):
                issues.append(f"私密约束“{constraint.id}”出现在公共信息中")
                break

    locked_ids = {item.id for item in package.brief.constraints if item.locked and item.content}
    covered_ids = set(package.world.covered_constraint_ids)
    for fact in package.world.facts:
        covered_ids.update(fact.covered_constraint_ids)
    for character in package.characters:
        covered_ids.update(character.covered_constraint_ids)
    missing_ids = sorted(locked_ids - covered_ids)
    if missing_ids:
        issues.append(f"以下用户硬约束未声明覆盖：{', '.join(missing_ids)}")
    return issues


def agents_from_character_specs(characters: list[CharacterSpec]):
    from .models import AbilityState, AgentState

    agents = []
    for index, character in enumerate(characters, start=1):
        knowledge = "\n".join(
            f"{key}：{value}" for key, value in character.knowledge.items()
        )
        agents.append(
            AgentState(
                name=character.name,
                profile=character.to_markdown(index),
                public_profile=character.public_profile,
                goals=list(character.goals),
                private_memory=[f"我的初始知识边界：{knowledge}"] if knowledge else [],
                relationships=dict(character.relationships),
                role=character.role,
                abilities=[ability.name for ability in character.abilities],
                id=character.id,
                faction=character.faction,
                current_location=character.initial_location,
                resources=dict(character.resources),
                ability_states={
                    ability.id: AbilityState(
                        id=ability.id,
                        name=ability.name,
                        action_type=ability.action_type,
                        description=ability.description,
                        phases=list(ability.phases),
                        uses_remaining=ability.uses,
                        target_scope=ability.target_scope,
                        effects=[asdict(effect) for effect in ability.effects],
                        visibility=list(ability.visibility),
                    )
                    for ability in character.abilities
                },
                goal_status={goal: "active" for goal in character.goals},
                known_facts={fact_id: "" for fact_id in character.known_fact_ids},
                false_beliefs=list(character.false_beliefs),
            )
        )
    return agents


def fallback_opening_scene(user_prompt: str) -> str:
    summary = " ".join((user_prompt or "").split())[:100]
    return f"围绕“{summary or '用户设定'}”的事件即将开始，所有参与者已经到场。"
