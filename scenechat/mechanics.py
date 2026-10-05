"""Versioned, declarative effects. No model-provided code is executed."""
from copy import copy, deepcopy

from .scenario import VALID_PATCH_OPERATIONS
from .role_selectors import matches_role
from .visibility import ViewerContext, can_access


TARGET_SCOPES = {"none", "self", "same_location", "any_active", "character", "location", "object", "proposal"}
TEMPLATES = {"", "legacy_tabletop"}
EFFECT_PLACEHOLDERS = {"$actor", "$target", "$value"}
BEAT_STATE_CONDITIONS = {"world_equals", "entity_equals", "agent_location", "goal_equals"}


def records_vote(rule):
    """A vote rule must record ballots rather than impersonate a win condition."""
    if rule.action_type != "vote":
        return True
    return (getattr(rule, "behavior_template", "") == "legacy_tabletop"
            or getattr(rule, "effect_mode", "rule") == "ability"
            or any(effect.op == "record_vote" for effect in rule.effects))


def validate_world_structure(world):
    """Catch cast-independent V2 mistakes before the expensive character call.

    Character-role reachability remains a package check after the cast exists.
    This does not silently rewrite a requested game mechanic.
    """
    if world.execution_version != 2:
        return []
    issues = []
    phases = {phase.name: phase for phase in world.phase_specs}
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
    for rule in world.rules:
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
            issues.append(f"规则 {rule.id} 没有允许其执行的角色行动阶段；event_only 只能由环境事件结算")
        for effect in rule.effects:
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
    return list(dict.fromkeys(issues))


def visible_entities(state, actor):
    viewer = ViewerContext(name=actor.name, role=actor.role, location=actor.current_location)
    return {key: {"kind": spec.get("kind"), "state": state.entity_states.get(key, {})}
            for key, spec in getattr(state.world_spec, "entities", {}).items()
            if can_access(spec.get("visibility"), viewer)}


def action_context(state, actor):
    viewer = ViewerContext(name=actor.name, role=actor.role, location=actor.current_location)
    phase = state.phase_specs.get(state.current_phase)
    allowed = set(phase.allowed_action_types) if phase and phase.allowed_action_types else None
    visible = visible_entities(state, actor)
    lines = []
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
        lines.append(f"- rule_id={rule.id}；action_type={rule.action_type}；目标类型={scope}{target_hint}；{rule.description}")
    return "\n".join(lines) or "- 无额外场景行动"


def validate_mechanics(package):
    world = package.world
    issues = []
    if world.execution_version not in {1, 2}:
        return ["execution_version 仅支持 1 或 2"]
    if world.execution_version == 1:
        return []
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

    if any(needs_elimination(rule) for rule in world.termination_rules):
        can_eliminate = any(
            any(effect.op == "set_agent_status" and effect.key in {"alive", "active"}
                and effect.value is False for effect in spec.effects)
            or (getattr(spec, "behavior_template", "") == "legacy_tabletop"
                and spec.action_type in {"vote", "eliminate", "poison"})
            for spec in specs
        )
        if not can_eliminate:
            issues.append("结束规则 faction_eliminated 要求角色淘汰，但规则和能力均无可执行的淘汰效果；"
                          "settle_votes 只更新提案状态，不会淘汰角色")
    effects = [effect for spec in specs for effect in spec.effects]

    def has_live_dependency(rule):
        """A phase gate alone cannot turn an immutable initial value into a result."""
        kind = rule.kind
        if kind in {"all_of", "any_of"}:
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

    for terminal in world.termination_rules:
        if not has_live_dependency(terminal) and terminal.kind != "manual":
            issues.append(
                f"结束规则 {terminal.id} 只依赖不会被行动或导演更新的初始状态；"
                "进入结算阶段不会使胜负条件变真。请连接可执行的状态效果，"
                "或改用实际存活角色的 faction_eliminated/faction_parity 等条件"
            )
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
    def check_end(rule):
        if rule.kind == "entity_equals":
            entity = world.entities.get(rule.target, {})
            if not isinstance(entity, dict) or not isinstance(entity.get("state"), dict) or rule.key not in entity["state"] or type(rule.value) is not type(entity["state"].get(rule.key)):
                issues.append(f"结束规则 {rule.id} 的实体状态引用或类型无效")
        for child in rule.conditions:
            check_end(child)
    for rule in world.termination_rules:
        check_end(rule)
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
