"""Closed, typed execution prerequisites; never evaluate generated code."""
from copy import deepcopy
import json


FIELDS = {
    'world_equals': {'kind', 'key', 'value'},
    'entity_equals': {'kind', 'target', 'key', 'value'},
    'agent_location': {'kind', 'target', 'value'},
    'all_active_at_location': {'kind', 'location'},
}


def parse_requirements(raw):
    if not isinstance(raw, list) or len(raw) > 32:
        raise ValueError('requires 必须是最多 32 项类型化前提数组')
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get('kind'), str) or item['kind'] not in FIELDS:
            raise ValueError('requires 仅支持 world_equals/entity_equals/agent_location/all_active_at_location')
        if set(item) != FIELDS[item['kind']]:
            raise ValueError('requires 字段不完整或包含表达式/脚本/未知字段')
        for key in FIELDS[item['kind']] - {'value'}:
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError('requires 引用必须是非空字符串')
    return deepcopy(raw)


def validate_requirements(world, spec):
    try:
        items = parse_requirements(getattr(spec, 'requires', []))
    except ValueError as exc:
        return [f'规则 {spec.id} requires 无效：{exc}']
    issues = []
    for index, item in enumerate(items):
        kind = item['kind']
        reasons = []
        if kind == 'world_equals':
            key = item['key']
            field = world.state_schema.get(key)
            if field is None:
                reasons.append(f'/world/state_schema/{key} 未声明；需引用真实字段或补全一致的状态声明')
            if key not in world.initial_state:
                reasons.append(f'/world/initial_state/{key} 缺少初始值'
                               + (f'（schema 类型 {field.value_type}）' if field else '')
                               + '；保留真实前提，按开场设定补值，不以删前提或改成另一条件掩盖缺失')
            else:
                expected_type = type(world.initial_state[key])
                if type(item['value']) is not expected_type:
                    reasons.append(f'value 类型 {type(item["value"]).__name__} 与初始字段类型 '
                                   f'{expected_type.__name__} 不一致，不转换真假/数值为字符串')
            if field is not None and not field.accepts_value(item['value']):
                reasons.append(f'value 不满足字段 {key} 的 {field.value_type} schema/allowed_values')
        elif kind == 'entity_equals':
            entities = ([(item['target'], world.entities.get(item['target']))] if item['target'] != '$target' else
                        [(name,e) for name,e in world.entities.items() if isinstance(e, dict) and e.get('kind') == spec.target_scope])
            if not entities:
                reasons.append(f'$target 无 {spec.target_scope} 范围内的声明实体')
            for name, entity in entities:
                if not isinstance(entity, dict) or not isinstance(entity.get('state'), dict) or item['key'] not in entity['state']:
                    reasons.append(f'/world/entities/{name}/state/{item["key"]} 未声明；人物位置不能冒充实体字段')
                elif type(item['value']) is not type(entity['state'][item['key']]):
                    reasons.append(f'value 类型 {type(item["value"]).__name__} 与实体 {name} 字段 '
                                   f'{item["key"]} 的 {type(entity["state"][item["key"]]).__name__} 不一致')
        elif kind == 'agent_location':
            # Only this actor or this action's character target. No arbitrary
            # character lookup whose presence is unknown at world-generation.
            if item['target'] not in {'$actor', '$target'}:
                reasons.append('行动前提人物 target 只支持 $actor 或人物目标 $target，不支持在这里填固定人物姓名')
            if item['value'] not in world.locations:
                reasons.append('value 必须是 /world/locations 中的地点原文')
            if item['target'] == '$target' and spec.target_scope not in {'self', 'same_location', 'any_active', 'character'}:
                reasons.append(f'$target 需要人物目标范围，当前 target_scope={spec.target_scope}')
        else:
            if item['location'] not in world.locations:
                reasons.append('location 必须是 /world/locations 中的地点原文')
        if reasons:
            issues.append(f'规则 {spec.id} requires/{index} 无效：' + '；'.join(reasons[:8]))
    return issues


def requirements_met(state, spec, intent):
    if validate_requirements(state.world_spec, spec):
        return False
    for item in spec.requires:
        kind = item['kind']
        if kind == 'world_equals':
            values, key = state.world_state, item['key']
        elif kind == 'entity_equals':
            target = intent.target if item['target'] == '$target' else item['target']
            values, key = state.entity_states.get(target, {}), item['key']
        elif kind == 'agent_location':
            target = intent.actor if item['target'] == '$actor' else intent.target
            agent = state.agents.get(target)
            if agent is None or agent.current_location != item['value']:
                return False
            continue
        else:
            active = [a for a in state.agents.values() if a.eligible]
            if not active or any(a.current_location != item['location'] for a in active):
                return False
            continue
        if key not in values or type(values[key]) is not type(item['value']) or values[key] != item['value']:
            return False
    return True


def visible_requirement_items(state, actor, spec):
    """Show only visible interface predicates, not their truth or private data."""
    from .visibility import ViewerContext, can_access
    viewer = ViewerContext(name=actor.name, role=actor.role, location=actor.current_location)
    shown = []
    for item in spec.requires:
        kind = item.get('kind')
        if kind == 'world_equals':
            field = state.state_schema.get(item.get('key'))
            allowed = bool(field and can_access(field.visibility, viewer))
        elif kind == 'entity_equals':
            entities = ([state.world_spec.entities.get(item.get('target'))] if item.get('target') != '$target' else
                        [e for e in state.world_spec.entities.values() if isinstance(e, dict) and e.get('kind') == spec.target_scope])
            allowed = bool(entities) and all(isinstance(e, dict) and can_access(e.get('visibility'), viewer) for e in entities)
        else:
            allowed = kind in {'agent_location', 'all_active_at_location'}
        if allowed:
            shown.append(item)
    return shown


def visible_requirements(state, actor, spec):
    shown = visible_requirement_items(state, actor, spec)
    return json.dumps(shown, ensure_ascii=False) if shown else '存在执行前提；不公开受限状态'


def failure_details(state, actor, spec, intent, *, target_selected=True):
    """Ground retry feedback in visible actual state, never private predicates."""
    from types import SimpleNamespace
    if validate_requirements(state.world_spec, spec):
        return ''
    failed = []
    for item in visible_requirement_items(state, actor, spec):
        if not target_selected and item.get('target') == '$target':
            continue
        single = SimpleNamespace(id=spec.id, target_scope=spec.target_scope, requires=[item])
        if requirements_met(state, single, intent):
            continue
        kind = item['kind']
        current = None
        if kind == 'agent_location':
            target = intent.actor if item['target'] == '$actor' else intent.target
            agent = state.agents.get(target)
            # Arbitrary other character locations never become public merely
            # because this rule asks about a character target.
            if agent is not None and (target == actor.name or agent.current_location == actor.current_location):
                current = agent.current_location
            else:
                continue
        elif kind == 'world_equals':
            current = state.world_state.get(item['key'])
        elif kind == 'entity_equals':
            target = intent.target if item['target'] == '$target' else item['target']
            current = state.entity_states.get(target, {}).get(item['key'])
        else:
            # Do not expose globally queried characters or their locations.
            continue
        failed.append({'requires': item, 'current': current})
    return json.dumps(failed, ensure_ascii=False) if failed else ''
