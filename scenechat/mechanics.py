"""Versioned, declarative effects. No model-provided code is executed."""
from copy import copy, deepcopy

from .scenario import VALID_PATCH_OPERATIONS
from .role_selectors import matches_role
from .visibility import ViewerContext, can_access


TARGET_SCOPES = {"none", "self", "same_location", "any_active", "character", "location", "object", "proposal"}
TEMPLATES = {"", "legacy_tabletop"}
EFFECT_PLACEHOLDERS = {"$actor", "$target", "$value"}
BEAT_STATE_CONDITIONS = {"world_equals", "entity_equals", "agent_location", "goal_equals"}


def _fixed_world_effect_issue(world, spec, effect):
    if effect.op != 'set_world' or (isinstance(effect.value, str) and effect.value in EFFECT_PLACEHOLDERS):
        return ''
    field = world.state_schema.get(effect.key)
    if field is None or not effect.key or effect.key.startswith('_') or not field.accepts_value(effect.value):
        return (f'规则/能力 {spec.id} 的 set_world({effect.key}, {effect.value!r}) '
                f'不能写入 state_schema 的类型/allowed_values；该固定效果将永远被运行时拒绝。'
                '请按用户机制修复效果与 schema 的对应关系，不让演员重试不可执行的值')
    return ''


def _string_placeholder_type_issue(world, spec, effect):
    """All three supported substitutions produce strings, never booleans/JSON.

    Catch fixed and scoped entity destinations before actors attempt an
    impossible action; do not coerce a spoken expectation into a result.
    """
    if not isinstance(effect.value, str) or effect.value not in EFFECT_PLACEHOLDERS:
        return ""
    mismatch = False
    if effect.op == "set_entity":
        entities = ([world.entities.get(effect.target)] if effect.target != "$target" else
                    [entity for entity in world.entities.values()
                     if isinstance(entity, dict) and entity.get("kind") == spec.target_scope])
        mismatch = any(isinstance(entity, dict) and isinstance(entity.get("state"), dict)
                       and effect.key in entity["state"] and not isinstance(entity["state"][effect.key], str)
                       for entity in entities)
    elif effect.op == "set_world":
        field = world.state_schema.get(effect.key)
        mismatch = bool(field and field.value_type not in {"string", "enum", "any"})
    if mismatch:
        return (f"规则/能力 {spec.id} 的 {effect.op}.{effect.key} 使用字符串占位符 {effect.value} "
                "写入非字符串状态；$value 是 Intent.expected_effect 文本，不会转成布尔、数字或 null。"
                "请按各合法选择声明固定同类型效果，不得要求演员反复提交无法执行的值")
    return ""


def records_vote(rule):
    """A vote rule must record ballots rather than impersonate a win condition."""
    if rule.action_type != "vote":
        return True
    return (getattr(rule, "behavior_template", "") == "legacy_tabletop"
            or getattr(rule, "effect_mode", "rule") == "ability"
            or any(effect.op == "record_vote" for effect in rule.effects))


def _entity_termination_issues(world):
    """Diagnose actual nested addresses, not synthetic child IDs like termination-1."""
    issues = []

    def check(rule, path, root_id):
        if rule.kind == "entity_equals":
            entity = world.entities.get(rule.target, {})
            fields = entity.get("state", {}) if isinstance(entity, dict) else {}
            if (not isinstance(fields, dict) or rule.key not in fields
                    or type(rule.value) is not type(fields.get(rule.key))):
                issues.append(
                    f"结束规则 {root_id} 的实体状态引用或类型无效：{path}，"
                    f"kind=entity_equals, target={rule.target}, key={rule.key}, value={rule.value!r}。"
                    "target 必须是 world.entities 中的真实物品/提案 ID，不是人物姓名或 CharacterSpec.id；"
                    f"现有实体={list(world.entities)}，该实体状态键={list(fields) if isinstance(fields, dict) else []}。"
                    "不能只把人物姓名改成人物 id；人物位置用 all_active_at_location 等支持的条件。"
                    "仅系统自行补出的、没有执行机制的失败分支可移除；用户明确要求的条件必须保留并连接真实机制"
                )
        for index, child in enumerate(rule.conditions):
            check(child, f"{path}/conditions/{index}", root_id)

    for index, terminal in enumerate(world.termination_rules):
        check(terminal, f"/world/termination_rules/{index}", terminal.id)
    return issues


def validate_world_structure(world):
    """Catch cast-independent V2 mistakes before the expensive character call.

    Character-role reachability remains a package check after the cast exists.
    This does not silently rewrite a requested game mechanic.
    """
    if world.execution_version != 2:
        return []
    issues = []
    phases = {phase.name: phase for phase in world.phase_specs}
    from .action_requirements import validate_requirements
    for rule in world.rules:
        issues.extend(validate_requirements(world, rule))
    for index, phase in enumerate(world.phase_specs):
        if not phase.completion_action_types:
            continue
        if phase.event_only or phase.advance_when != "all_eligible_acted":
            issues.append(f"/world/phase_specs/{index}/completion_action_types 仅用于 all_eligible_acted 角色阶段，不能用于环境、手动或投票阶段")
        missing = set(phase.completion_action_types) - set(phase.allowed_action_types)
        if phase.allowed_action_types and missing:
            issues.append(f"/world/phase_specs/{index}/completion_action_types 包含本阶段不允许的动作：{sorted(missing)}")
    for phase in phases.values():
        if phase.advance_when == "all_active_voted" and any(
            action in phase.allowed_action_types for action in ("pass", "observe")
        ):
            issues.append(
                f"投票阶段“{phase.name}”允许不记录选票的 pass/observe，"
                "却要求 all_active_voted；必须实现真正的弃权票契约或移除这些行动"
            )
        if phase.advance_when == "manual" and phase.next_phase:
            exit_rules = [
                rule for rule in world.rules
                if (not rule.phases or phase.name in rule.phases)
                and (not phase.allowed_action_types or rule.action_type in phase.allowed_action_types)
                and any(effect.op == "set_phase" and effect.value == phase.next_phase for effect in rule.effects)
            ]
            if not exit_rules:
                issues.append(f"手动阶段“{phase.name}”没有可执行的 set_phase 出口")
    for rule_index, rule in enumerate(world.rules):
        if rule.action_type != "vote" and any(effect.op == "record_vote" for effect in rule.effects):
            issues.append(f"规则 {rule.id} 不是 vote 行动却记录选票；提问、质疑等不能暗中投票")
        if not records_vote(rule):
            issues.append(f"规则 {rule.id} 声明 vote 却不记录选票；胜负请放在 termination_rules，不能用投票行动无条件设置胜负")
        selected = [phases[name] for name in rule.phases if name in phases] if rule.phases else list(phases.values())
        if selected and not any(
            not phase.event_only
            and (not phase.allowed_action_types or rule.action_type in phase.allowed_action_types)
            for phase in selected
        ):
            actor_phases = [phase for phase in selected if not phase.event_only]
            if not actor_phases:
                issues.append(f"规则 {rule.id} 没有允许其执行的角色行动阶段；它仅绑定 event_only 环境阶段，不能安排角色执行。检查 /world/rules/{rule_index}/phases 与效果触发时机")
            else:
                pointers = [f"/world/phase_specs/{index}/allowed_action_types={phase.allowed_action_types}"
                            for index, phase in enumerate(world.phase_specs) if phase in actor_phases]
                issues.append(f"规则 {rule.id} 没有允许其执行的角色行动阶段：action_type={rule.action_type} "
                              f"未列入其非 event_only 角色阶段的 allowed_action_types；{'；'.join(pointers)}。"
                              f"保留此动作时须同步阶段许可与 /world/rules/{rule_index}/action_type；"
                              "若阶段已自动推进而该规则只重复同一 set_phase 出口，可删除冗余规则，不能重复返回原规则")
        for effect in rule.effects:
            fixed_issue = _fixed_world_effect_issue(world, rule, effect)
            if fixed_issue:
                issues.append(fixed_issue)
            placeholder_issue = _string_placeholder_type_issue(world, rule, effect)
            if placeholder_issue:
                issues.append(placeholder_issue)
            for value in (effect.key, effect.target, effect.value):
                if isinstance(value, str) and value.startswith("$") and value not in EFFECT_PLACEHOLDERS:
                    issues.append(f"规则/能力 {rule.id} 使用不支持的效果占位符 {value}")
            if effect.op not in VALID_PATCH_OPERATIONS:
                issues.append(f"规则/能力 {rule.id} 包含不支持的操作 {effect.op}")
            if effect.op == "set_entity" and effect.target in world.entities:
                entity = world.entities[effect.target]
                state = entity.get("state") if isinstance(entity, dict) else None
                if (isinstance(state, dict) and effect.key in state
                        and not (isinstance(effect.value, str) and effect.value in EFFECT_PLACEHOLDERS)
                        and type(effect.value) is not type(state[effect.key])):
                    issues.append(f"规则/能力 {rule.id} 的实体状态值类型不匹配")
    for beat in world.beat_specs:
        if beat.completion_mode == "state" and not beat.completion_conditions:
            issues.append(f"节点 {beat.id} 必须声明 completion_conditions")
        for condition in beat.completion_conditions:
            if not isinstance(condition, dict) or condition.get("kind") not in BEAT_STATE_CONDITIONS:
                issues.append(f"节点 {beat.id} 使用不支持的完成条件；若无可声明的状态结果，请使用 narrative 与已提交事件证据")
    # Diagnose an ordinary conversation action hidden by an unconditional
    # special effect. Never rewrite priority (it is also an authorization
    # boundary), and do not infer executable conditions from descriptions.
    for ordinary in world.rules:
        if (ordinary.action_type not in {"speak", "act", "observe", "pass"}
                or ordinary.effects or ordinary.behavior_template
                or ordinary.effect_mode != "rule"):
            continue
        for special in world.rules:
            if (special.id == ordinary.id or not special.effects
                    or special.action_type != ordinary.action_type
                    or special.target_scope != ordinary.target_scope
                    or special.priority <= ordinary.priority
                    or special.allowed_roles != ordinary.allowed_roles):
                continue
            shared = [name for name, phase in phases.items()
                      if not phase.event_only
                      and (not ordinary.phases or name in ordinary.phases)
                      and (not special.phases or name in special.phases)
                      and (not phase.allowed_action_types or ordinary.action_type in phase.allowed_action_types)]
            if shared:
                issues.append(
                    f"规则 {ordinary.id} 的普通 {ordinary.action_type} 在阶段 {'、'.join(shared)} "
                    f"被高优先级规则 {special.id} 的无条件效果遮蔽；description 不是执行条件。"
                    "请为有意选择的特殊操作使用独立 action_type 并同步阶段动作，"
                    "或保留明确的统一动作效果并移除重复普通规则；不得降低优先级绕过限制"
                )
    issues.extend(_entity_termination_issues(world))
    return list(dict.fromkeys(issues))


def visible_entities(state, actor):
    viewer = ViewerContext(name=actor.name, role=actor.role, location=actor.current_location)
    return {key: {"kind": spec.get("kind"), "state": state.entity_states.get(key, {})}
            for key, spec in getattr(state.world_spec, "entities", {}).items()
            if can_access(spec.get("visibility"), viewer)}


def _visible_rule_effects(state, actor, rule):
    """Executable consequences visible to this actor, never private lookups.

    These are prospective interface semantics, not observations or new facts.
    Unknown operation visibility is omitted rather than inferred from prose.
    """
    from dataclasses import asdict
    viewer = ViewerContext(name=actor.name, role=actor.role, location=actor.current_location)
    visible = visible_entities(state, actor)
    output = []
    for effect in rule.effects:
        allowed = False
        if effect.op in {"set_world", "increment_world"}:
            field = state.state_schema.get(effect.key)
            allowed = bool(field and can_access(field.visibility, viewer))
        elif effect.op == "set_entity":
            targets = ([visible.get(effect.target)] if effect.target != "$target" else
                       [entity for entity in visible.values() if entity.get("kind") == rule.target_scope])
            allowed = any(entity and effect.key in entity.get("state", {}) for entity in targets)
        elif effect.op == "move_agent":
            allowed = effect.target in {actor.name, "$actor"} and effect.value in state.locations
        elif effect.op in {"consume_resource", "set_resource"}:
            allowed = effect.target in {actor.name, "$actor"} and effect.key in actor.resources
        elif effect.op == "set_phase":
            allowed = effect.value in state.phase_specs
        elif effect.op in {"record_vote", "clear_votes"}:
            allowed = True
        if allowed:
            operation = asdict(effect)
            relevant = {
                "set_world": ("key", "value"), "increment_world": ("key", "amount"),
                "set_entity": ("target", "key", "value"), "move_agent": ("target", "value"),
                "consume_resource": ("target", "key", "amount"),
                "set_resource": ("target", "key", "value"), "set_phase": ("value",),
                "record_vote": ("target",), "clear_votes": (),
            }[effect.op]
            output.append({key: operation[key] for key in ("op", *relevant)})
    return output


def action_context(state, actor):
    viewer = ViewerContext(name=actor.name, role=actor.role, location=actor.current_location)
    phase = state.phase_specs.get(state.current_phase)
    allowed = set(phase.allowed_action_types) if phase and phase.allowed_action_types else None
    visible = visible_entities(state, actor)
    lines = []
    if phase and phase.completion_action_types:
        lines.append("本阶段的实际参与仅登记 action_type=" + "、".join(phase.completion_action_types)
                     + "。其他获准交流可以进行，但不代替这一阶段所需的实际提交；完成不由台词宣告。")
    for rule in state.rules:
        if ((rule.phases and state.current_phase not in rule.phases)
                or (allowed is not None and rule.action_type not in allowed)
                or (getattr(state.world_spec, "execution_version", 1) == 2 and not records_vote(rule))
                or not matches_role(actor.role, rule.allowed_roles)
                or not can_access(rule.visibility, viewer)):
            continue
        scope = rule.target_scope
        if scope in {"character", "any_active"}:
            targets = [item.name for item in state.agents.values() if item.eligible]
        elif scope == "same_location":
            targets = [item.name for item in state.agents.values()
                       if item.eligible and item.current_location == actor.current_location]
        elif scope in {"object", "proposal"}:
            targets = [key for key, item in visible.items() if item.get("kind") == scope]
        elif scope == "location":
            targets = list(state.locations)
        elif scope == "self":
            targets = [actor.name]
        else:
            targets = []
        target_hint = f"；可填 target={','.join(targets[:12]) or '无有效目标'}" if scope != "none" else "；target 留空"
        lines.append(f"- rule_id={rule.id}；action_type={rule.action_type}；priority={rule.priority}；目标类型={scope}{target_hint}；{rule.description}")
        if getattr(rule, 'requires', []):
            from .action_requirements import visible_requirements, failure_details
            from types import SimpleNamespace
            lines.append('  执行前提（全部满足；不表示已经满足）：' + visible_requirements(state, actor, rule))
            unavailable = failure_details(state, actor, rule, SimpleNamespace(actor=actor.name, target=''), target_selected=False)
            if unavailable:
                lines.append('  当前可见状态不满足此前提，不能重复提交该动作：' + unavailable)
        if getattr(state.world_spec, "execution_version", 1) == 2 and rule.effect_mode in {"rule", "stack"}:
            consequences = _visible_rule_effects(state, actor, rule)
            if consequences:
                import json
                lines.append("  提交该动作后引擎将登记的可见效果（目前尚未发生）："
                             + json.dumps(consequences, ensure_ascii=False, separators=(",", ":"))
                             + "。公开动作须与本次实际提交相符；仅准备或走向时不提交表示已经到达/完成的接口，尝试型效果不保证成功。")
        if getattr(state.world_spec, "execution_version", 1) == 2 and rule.effect_mode in {"ability", "stack"}:
            from types import SimpleNamespace
            from .scenario import PatchTemplate
            import json
            for ability in actor.ability_states.values():
                if (ability.action_type not in {rule.action_type, "act"}
                        or (ability.phases and state.current_phase not in ability.phases)
                        or ability.uses_remaining == 0):
                    continue
                source = SimpleNamespace(
                    effects=[PatchTemplate.from_mapping(effect) for effect in ability.effects],
                    target_scope=ability.target_scope)
                consequences = _visible_rule_effects(state, actor, source)
                lines.append(f"  本人执行来源：ability={ability.id}；action_type={ability.action_type}"
                             + ("；提交该能力后引擎将登记的可见效果（目前尚未发生）："
                                + json.dumps(consequences, ensure_ascii=False, separators=(",", ":"))
                                if consequences else "；效果仍按本人能力与可见权限执行，不由公共规则代办")
                             + "。只能代表本人的选择，不能替其他人确认。")
    if lines and getattr(state.world_spec, "execution_version", 1) == 2:
        lines.append("同一动作的角色、阶段和目标均匹配时，执行最高 priority 的规则；不能用较低优先级 rule_id 绕过。description 不会额外判断条件。")
    return "\n".join(lines) or "- 无额外场景行动"


def _potential_execution_effects(package):
    """Exclude inert sources, without claiming to prove all semantic reachability.

    Different target scopes may create distinct legal routes, so uncertain
    overlaps remain conservative. Exact-scope priority follows the Resolver.
    Resource costs execute even under an authoritative rule-only effect mode.
    """
    world = package.world
    effects = [effect for rule in world.rules if rule.effect_mode in {"rule", "stack"}
               for effect in rule.effects]
    for character in package.characters:
        for ability in character.abilities:
            if ability.uses == 0:
                continue
            phases = [phase for phase in world.phase_specs
                      if not phase.event_only and matches_role(character.role, phase.actor_roles)
                      and (not ability.phases or phase.name in ability.phases)
                      and (not phase.allowed_action_types or ability.action_type in phase.allowed_action_types
                           or (ability.action_type == "act" and any(
                               rule.action_type in phase.allowed_action_types
                               and (not rule.phases or phase.name in rule.phases)
                               and matches_role(character.role, rule.allowed_roles)
                               and rule.effect_mode in {"ability", "stack"} for rule in world.rules)))]
            # Legacy/incomplete fixtures without phase specs remain an
            # overapproximation; other validators diagnose missing phases.
            active = not world.phase_specs
            for phase in phases:
                if ability.action_type == "act":
                    # Resolver accepts a generic act ability for a declared
                    # special action. Preserve this legacy capability route.
                    active = True
                    continue
                rules = [rule for rule in world.rules if rule.action_type == ability.action_type
                         and (not rule.phases or phase.name in rule.phases)
                         and matches_role(character.role, rule.allowed_roles)]
                if not rules or any(rule.target_scope != ability.target_scope for rule in rules):
                    active = True
                elif max(rules, key=lambda rule: rule.priority).effect_mode in {"ability", "stack"}:
                    active = True
            if phases or not world.phase_specs:
                effects.extend(effect for effect in ability.effects if active or effect.op == "consume_resource")
    return effects


def validate_mechanics(package):
    world = package.world
    issues = []
    if world.execution_version not in {1, 2}:
        return ["execution_version 仅支持 1 或 2"]
    if world.execution_version == 1:
        return []
    issues.extend(validate_world_structure(world))
    from .scenario_checks import rule_actors
    for index, phase in enumerate(world.phase_specs):
        if not phase.completion_action_types:
            continue
        for character in package.characters:
            if not matches_role(character.role, phase.actor_roles):
                continue
            generic = set(phase.completion_action_types).intersection({"speak", "act", "observe", "pass"})
            usable = any(
                rule.action_type in phase.completion_action_types
                and character in rule_actors(package, rule, phase)
                and can_access(rule.visibility, ViewerContext(name=character.name, role=character.role,
                                                              location=character.initial_location))
                and (rule.effect_mode in {"rule", "stack"} or any(
                    ability.action_type in {rule.action_type, "act"} and ability.uses_remaining != 0
                    and (not ability.phases or phase.name in ability.phases)
                    for ability in character.abilities))
                for rule in world.rules)
            if not generic and not usable:
                issues.append(f"/world/phase_specs/{index}/completion_action_types 中角色 {character.name} "
                              "没有可执行的参与动作；补齐本人权限/能力或调整 actor_roles，不能以发言代办实际提交")
    if world.audience_policy not in {"limited", "omniscient"} or world.reveal_policy not in {"preserve_suspense", "allow_reveal"}:
        issues.append("观众策略必须使用 limited/omniscient 和 preserve_suspense/allow_reveal")
    for entity_id, entity in world.entities.items():
        if not isinstance(entity, dict) or entity.get("kind") not in {"object", "proposal"} or not isinstance(entity.get("state"), dict):
            issues.append(f"实体 {entity_id} 必须声明 kind=object/proposal 和 state 对象")
        if entity_id in {c.name for c in package.characters} | set(world.locations):
            issues.append(f"实体 ID {entity_id} 与角色或地点重名")
    specs = list(world.rules) + [a for c in package.characters for a in c.abilities]
    for spec in specs:
        if spec in world.rules and not records_vote(spec):
            issues.append(f"规则 {spec.id} 声明 vote 却不记录选票；胜负请放在 termination_rules")
        if spec.target_scope not in TARGET_SCOPES:
            issues.append(f"规则/能力 {spec.id} 使用不支持的 target_scope")
        if getattr(spec, "effect_mode", "rule") not in {"rule", "ability", "stack"}:
            issues.append(f"规则 {spec.id} effect_mode 仅支持 rule/ability/stack")
        if getattr(spec, "behavior_template", "") not in TEMPLATES:
            issues.append(f"规则 {spec.id} 使用不支持的 behavior_template")
        if getattr(spec, "behavior_template", "") and spec.effects and spec.effect_mode != "stack":
            issues.append(f"规则 {spec.id} 的模板和显式效果叠加需要 effect_mode=stack")
        if getattr(spec, "behavior_template", "") and spec.target_scope in {"object", "proposal", "location"} and spec.action_type != "move":
            issues.append(f"规则 {spec.id} 的 legacy_tabletop 模板仅用于人物目标")
        if spec.action_type not in {"speak", "act", "observe", "pass"} and not spec.effects and not getattr(spec, "behavior_template", "") and spec in world.rules and spec.effect_mode != "ability":
            issues.append(f"规则 {spec.id} 没有可执行效果，请声明 effects、能力效果来源或显式模板")
        for effect in spec.effects:
            placeholder_issue = _string_placeholder_type_issue(world, spec, effect)
            if placeholder_issue:
                issues.append(placeholder_issue)
            unsupported = list(dict.fromkeys(
                value for value in (effect.key, effect.target, effect.value)
                if isinstance(value, str) and value.startswith("$") and value not in EFFECT_PLACEHOLDERS
            ))
            for value in unsupported:
                issues.append(f"规则/能力 {spec.id} 使用不支持的效果占位符 {value}")
            if effect.op not in VALID_PATCH_OPERATIONS:
                issues.append(f"规则/能力 {spec.id} 包含不支持的操作 {effect.op}")
            if effect.op in {"set_world", "increment_world"} and effect.key not in world.state_schema:
                issues.append(f"规则/能力 {spec.id} 必须为状态 {effect.key} 声明 state_schema")
            fixed_issue = _fixed_world_effect_issue(world, spec, effect)
            if fixed_issue:
                issues.append(fixed_issue)
            if effect.op == "set_entity" and effect.target not in {"$target"} | set(world.entities):
                issues.append(f"规则/能力 {spec.id} 引用了未知实体 {effect.target}")
            if effect.op in {"consume_resource", "consume_ability", "settle_votes"} and effect.amount <= 0:
                issues.append(f"规则/能力 {spec.id} 的扣费或投票阈值必须为正数")
            if effect.op == "set_agent_status" and (effect.key not in {"alive", "active"} or not isinstance(effect.value, bool)):
                issues.append(f"规则/能力 {spec.id} 的人物状态仅支持布尔 alive/active；生命值请使用 resources")
            if effect.op == "set_phase" and effect.value not in world.phases:
                issues.append(f"规则/能力 {spec.id} 引用未知目标阶段")
            if effect.op in {"set_entity", "settle_votes"}:
                candidates = [world.entities.get(effect.target)] if effect.target != "$target" else [e for e in world.entities.values() if isinstance(e, dict) and e.get("kind") == spec.target_scope]
                if not candidates or any(not isinstance(e, dict) or effect.key not in e.get("state", {}) for e in candidates):
                    issues.append(f"规则/能力 {spec.id} 的实体状态键 {effect.key} 未在目标中声明")
                elif any((not isinstance(e["state"][effect.key], bool)) if effect.op == "settle_votes" else (
                    not (isinstance(effect.value, str) and effect.value in EFFECT_PLACEHOLDERS)
                    and not unsupported and type(effect.value) is not type(e["state"][effect.key])
                ) for e in candidates):
                    issues.append(f"规则/能力 {spec.id} 的实体状态值类型不匹配")
    def needs_elimination(rule):
        return rule.kind == "faction_eliminated" or any(needs_elimination(child) for child in rule.conditions)

    effects = _potential_execution_effects(package)
    if any(needs_elimination(rule) for rule in world.termination_rules):
        can_eliminate = any(
            effect.op == "set_agent_status" and effect.key in {"alive", "active"}
            and effect.value is False for effect in effects
        ) or any(
            (getattr(spec, "behavior_template", "") == "legacy_tabletop"
                and spec.action_type in {"vote", "eliminate", "poison"})
            for spec in specs
        )
        if not can_eliminate:
            issues.append("结束规则 faction_eliminated 要求角色淘汰，但规则和能力均无可执行的淘汰效果；"
                          "settle_votes 只更新提案状态，不会淘汰角色")
    # Public effects on an ability-only rule are inert in the Resolver. They
    # cannot prove that an ending is reachable or that anyone has signed.
    for index, rule in enumerate(world.rules):
        if rule.effect_mode != "ability":
            continue
        holders = [character for character in package.characters
                   if matches_role(character.role, rule.allowed_roles)]
        if not any(ability.action_type in {rule.action_type, "act"} and ability.uses != 0
                   and any(not phase.event_only
                           and matches_role(character.role, phase.actor_roles)
                           and (not rule.phases or phase.name in rule.phases)
                           and (not ability.phases or phase.name in ability.phases)
                           and (not phase.allowed_action_types or rule.action_type in phase.allowed_action_types)
                           for phase in world.phase_specs)
                   for character in holders for ability in character.abilities):
            issues.append(f"/world/rules/{index} 的 effect_mode=ability 没有有权角色的同类型可用能力；"
                          "公共 effects 不执行，须为实际持有者声明能力并对接阶段许可")

    def already_satisfied_literal(rule):
        if rule.kind == "all_of":
            return bool(rule.conditions) and all(already_satisfied_literal(child) for child in rule.conditions)
        if rule.kind == "any_of":
            return any(already_satisfied_literal(child) for child in rule.conditions)
        if rule.kind == "world_equals":
            initial = world.initial_state.get(rule.key)
            return rule.key in world.initial_state and type(initial) is type(rule.value) and initial == rule.value
        if rule.kind == "entity_equals":
            entity = world.entities.get(rule.target)
            initial = entity.get("state") if isinstance(entity, dict) else None
            if not isinstance(initial, dict):
                return False
            return rule.key in initial and type(initial[rule.key]) is type(rule.value) and initial[rule.key] == rule.value
        return False

    def has_live_dependency(rule):
        """A phase gate alone cannot turn an immutable initial value into a result."""
        kind = rule.kind
        if kind == "all_of":
            return (any(has_live_dependency(child) for child in rule.conditions)
                    and all(has_live_dependency(child) or already_satisfied_literal(child)
                            for child in rule.conditions))
        if kind == "any_of":
            return any(has_live_dependency(child) for child in rule.conditions)
        if kind == "entity_equals":
            return any(
                effect.op in {"set_entity", "settle_votes"}
                and effect.key == rule.key
                and (effect.target == rule.target or effect.target == "$target")
                for effect in effects
            )
        if kind == "world_equals":
            if rule.key in {"phase", "current_phase", "round", "round_number"}:
                return True
            field = world.state_schema.get(rule.key)
            return any(effect.op in {"set_world", "increment_world"} and effect.key == rule.key
                       for effect in effects) or bool(field and "director" in field.mutable_by)
        if kind in {"faction_eliminated", "faction_parity"}:
            return any(effect.op == "set_agent_status" and effect.key in {"alive", "active"}
                       for effect in effects) or any(
                spec.behavior_template == "legacy_tabletop"
                and spec.action_type in {"vote", "eliminate", "poison"} for spec in specs
            )
        if kind == "all_goals_completed":
            return any(effect.op == "set_goal_status" for effect in effects)
        if kind == "all_active_at_location":
            return any(effect.op == "move_agent" for effect in effects)
        return False

    def check_required_dependencies(rule):
        if rule.kind != "all_of":
            return
        for child in rule.conditions:
            if not has_live_dependency(child) and not already_satisfied_literal(child):
                issues.append(
                    f"结束规则 {child.id} 的必需条件没有可执行来源："
                    f"kind={child.kind}, target={child.target}, key={child.key}；"
                    "all_of 要求每一项均可达到，不能只更新其中一人的状态就认为全体完成"
                )
            check_required_dependencies(child)

    for terminal in world.termination_rules:
        if not has_live_dependency(terminal) and terminal.kind != "manual":
            issues.append(
                f"结束规则 {terminal.id} 只依赖不会被行动或导演更新的初始状态；"
                "进入结算阶段不会使胜负条件变真。请连接可执行的状态效果，"
                "或改用实际存活角色的 faction_eliminated/faction_parity 等条件"
            )
        check_required_dependencies(terminal)
    for index, left in enumerate(world.rules):
        for right in world.rules[index + 1:]:
            phases_overlap = not left.phases or not right.phases or bool(set(left.phases) & set(right.phases))
            roles_overlap = any(matches_role(c.role, left.allowed_roles) and matches_role(c.role, right.allowed_roles) for c in package.characters)
            if left.action_type == right.action_type and left.priority == right.priority and phases_overlap and roles_overlap:
                issues.append(f"规则 {left.id}/{right.id} 匹配冲突，请设置不同 priority，禁止依赖列表顺序")
    for beat in world.beat_specs:
        if beat.completion_mode not in {"state", "narrative"}:
            issues.append(f"节点 {beat.id} completion_mode 仅支持 state/narrative")
        if beat.completion_mode == "state" and not beat.completion_conditions:
            issues.append(f"节点 {beat.id} 必须声明 completion_conditions")
        for condition in beat.completion_conditions:
            if not isinstance(condition, dict) or condition.get("kind") not in {"world_equals", "entity_equals", "agent_location", "goal_equals"}:
                issues.append(f"节点 {beat.id} 使用不支持的完成条件")
                continue
            kind = condition["kind"]
            target = condition.get("target")
            if kind == "world_equals" and condition.get("key") not in world.state_schema:
                issues.append(f"节点 {beat.id} 引用未知世界状态")
            if kind == "entity_equals" and target not in world.entities:
                issues.append(f"节点 {beat.id} 引用未知实体")
            if kind in {"agent_location", "goal_equals"} and target not in {c.name for c in package.characters}:
                issues.append(f"节点 {beat.id} 引用未知角色")
    issues.extend(_entity_termination_issues(world))
    return list(dict.fromkeys(issues))


def validate_patch(state, operations):
    """Preflight every operation against a shadow, including cumulative costs."""
    if len(operations) > 50:
        return "单次行动超过 50 个状态操作"
    shadow = copy(state)
    shadow.agents = deepcopy(state.agents)
    shadow.world_state = deepcopy(state.world_state)
    shadow.entity_states = deepcopy(state.entity_states)
    shadow.votes = dict(state.votes)
    shadow.protected_agents = set(state.protected_agents)
    shadow.phase_action_log = set(state.phase_action_log)
    for item in operations:
        op, key, target = item.get("op"), item.get("key"), item.get("target")
        value, amount = item.get("value"), item.get("amount", 1)
        agent = shadow.agents.get(target)
        valid = op in VALID_PATCH_OPERATIONS
        if op in {"consume_resource", "consume_ability", "increment_world"}:
            valid &= isinstance(amount, int) and not isinstance(amount, bool)
        if op in {"consume_resource", "consume_ability"}:
            valid &= isinstance(amount, int) and amount > 0
        if op in {"set_world", "increment_world"}:
            candidate = value
            if op == "increment_world":
                current = shadow.world_state.get(key, 0)
                valid &= isinstance(current, int) and not isinstance(current, bool)
                candidate = current + amount if valid else None
            valid &= shadow._valid_world_value(key, candidate)
        elif op == "set_entity":
            valid &= target in shadow.entity_states and key in shadow.entity_states.get(target, {})
            if valid:
                valid &= type(value) is type(shadow.entity_states[target][key])
        elif op in {"move_agent", "set_agent_status", "set_resource", "consume_resource", "consume_ability", "set_goal_status", "set_relationship", "add_known_fact", "protect_agent"}:
            valid &= agent is not None
            if agent is not None:
                if op == "move_agent": valid &= value in shadow.locations
                if op == "set_agent_status": valid &= key in {"alive", "active"} and isinstance(value, bool)
                if op == "set_resource": valid &= key in agent.resources and type(value) is type(agent.resources.get(key))
                if op == "consume_resource":
                    current = agent.resources.get(key)
                    valid &= isinstance(current, int) and isinstance(amount, int) and current >= amount
                if op == "consume_ability":
                    ability = agent.ability_states.get(key)
                    valid &= ability is not None and isinstance(amount, int) and (ability.uses_remaining is None or ability.uses_remaining >= amount)
                if op == "set_goal_status": valid &= key in agent.goal_status and value in {"pending", "active", "completed", "failed"}
                if op == "set_relationship": valid &= key in shadow.agents and isinstance(value, str)
                if op == "add_known_fact": valid &= bool(key) and isinstance(value, str)
        elif op == "record_vote":
            valid &= item.get("actor") in shadow.agents and (target in shadow.agents or target in shadow.entity_states)
        elif op == "settle_votes":
            valid &= isinstance(amount, int) and amount > 0 and target in shadow.entity_states and isinstance(shadow.entity_states.get(target, {}).get(key), bool)
        elif op == "set_phase":
            valid &= bool(shadow.phase_sequence) and (not value or value in shadow.phase_sequence)
        if not valid:
            return f"操作 {op} 的目标、值或资源余额无效；本次行动未提交任何变化"
        shadow.apply_state_patch([item])
    return ""


def patch_changes_state(state, operations):
    """Ignore no-op rewrites and bookkeeping when measuring story movement."""
    for op in operations:
        kind, key, target, value = op.get("op"), op.get("key"), op.get("target"), op.get("value")
        agent = state.agents.get(target)
        if kind == "set_world" and state.world_state.get(key) != value: return True
        if kind == "increment_world" and op.get("amount", 1): return True
        if kind == "set_entity" and state.entity_states.get(target, {}).get(key) != value: return True
        if kind == "move_agent" and agent and agent.current_location != value: return True
        if kind == "set_agent_status" and agent and getattr(agent, key, None) != value: return True
        if kind == "set_resource" and agent and agent.resources.get(key) != value: return True
        if kind == "consume_resource" and agent and op.get("amount", 1) and agent.resources.get(key, 0): return True
        if kind == "consume_ability" and agent and key in agent.ability_states and agent.ability_states[key].uses_remaining is not None: return True
        if kind == "set_goal_status" and agent and agent.goal_status.get(key) != value: return True
        if kind == "record_vote" and state.votes.get(op.get("actor")) != target: return True
        if kind == "add_known_fact" and agent and agent.known_facts.get(key) != value: return True
        if kind == "set_phase" and value and state.current_phase != value: return True
    return False
