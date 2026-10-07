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
        if not isinstance(item, dict) or item.get('kind') not in FIELDS:
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
        valid = True
        if kind == 'world_equals':
            valid = item['key'] in world.state_schema and item['key'] in world.initial_state
            if valid:
                valid = (type(item['value']) is type(world.initial_state[item['key']])
                         and world.state_schema[item['key']].accepts_value(item['value']))
        elif kind == 'entity_equals':
            entities = ([world.entities.get(item['target'])] if item['target'] != '$target' else
                        [e for e in world.entities.values() if isinstance(e, dict) and e.get('kind') == spec.target_scope])
            valid = bool(entities) and all(isinstance(e, dict) and isinstance(e.get('state'), dict)
                                         and item['key'] in e['state']
                                         and type(item['value']) is type(e['state'][item['key']]) for e in entities)
        elif kind == 'agent_location':
            # Only this actor or this action's character target. No arbitrary
            # character lookup whose presence is unknown at world-generation.
            valid = item['target'] in {'$actor', '$target'} and item['value'] in world.locations
            if item['target'] == '$target':
                valid = valid and spec.target_scope in {'self', 'same_location', 'any_active', 'character'}
        else:
            valid = item['location'] in world.locations
        if not valid:
            issues.append(f'规则 {spec.id} requires/{index} 引用、目标范围或值类型无效')
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
