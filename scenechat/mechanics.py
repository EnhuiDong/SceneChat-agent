"""Versioned, declarative effects. No model-provided code is executed."""
from copy import copy, deepcopy

from .scenario import VALID_PATCH_OPERATIONS
from .role_selectors import matches_role
from .visibility import ViewerContext, can_access


TARGET_SCOPES = {"none", "self", "same_location", "any_active", "character", "location", "object", "proposal"}
TEMPLATES = {"", "legacy_tabletop"}


def visible_entities(state, actor):
    viewer = ViewerContext(name=actor.name, role=actor.role, location=actor.current_location)
    return {key: {"kind": spec.get("kind"), "state": state.entity_states.get(key, {})}
            for key, spec in getattr(state.world_spec, "entities", {}).items()
            if can_access(spec.get("visibility"), viewer)}


def action_context(state, actor):
    viewer = ViewerContext(name=actor.name, role=actor.role, location=actor.current_location)
    return "\n".join(
        f"- action_type={rule.action_type}；目标类型={rule.target_scope}；{rule.description}"
        for rule in state.rules if (not rule.phases or state.current_phase in rule.phases)
        and matches_role(actor.role, rule.allowed_roles) and can_access(rule.visibility, viewer)
    ) or "- 无额外场景行动"


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
                elif any((not isinstance(e["state"][effect.key], bool)) if effect.op == "settle_votes" else (type(effect.value) is not type(e["state"][effect.key])) for e in candidates):
                    issues.append(f"规则/能力 {spec.id} 的实体状态值类型不匹配")
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
    return issues


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
        if kind == "set_goal_status" and agent and agent.goal_status.get(key) != value: return True
        if kind == "record_vote" and state.votes.get(op.get("actor")) != target: return True
        if kind == "add_known_fact" and agent and agent.known_facts.get(key) != value: return True
        if kind == "set_phase" and value and state.current_phase != value: return True
    return False
