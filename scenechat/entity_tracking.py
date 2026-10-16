"""Separate declared execution state from untracked opening decoration.

No prose-to-state inference, automatic patches, or changes to Resolver facts.
"""


def tracked_keys(state, entity_id):
    world = state.world_spec
    spec = getattr(world, 'entities', {}).get(entity_id, {})
    kind = spec.get('kind') if isinstance(spec, dict) else None
    keys = set()

    def effect(op):
        target = op.get('target')
        if op.get('op') in {'set_entity', 'settle_votes'} and target == entity_id:
            keys.add(op.get('key'))

    from dataclasses import asdict
    for rule in state.rules:
        for item in rule.effects:
            op = asdict(item)
            if op.get('target') == '$target' and rule.target_scope == kind:
                op['target'] = entity_id
            effect(op)
        for item in rule.requires:
            if item.get('kind') == 'entity_equals' and (
                item.get('target') == entity_id or item.get('target') == '$target' and rule.target_scope == kind):
                keys.add(item.get('key'))
    for actor in state.agents.values():
        for ability in actor.ability_states.values():
            for item in ability.effects:
                op = dict(item)
                if op.get('target') == '$target' and ability.target_scope == kind:
                    op['target'] = entity_id
                effect(op)

    def condition(item):
        if item.get('kind') == 'entity_equals' and item.get('target') == entity_id:
            keys.add(item.get('key'))
        for child in item.get('conditions') or []:
            if isinstance(child, dict):
                condition(child)

    for rule in getattr(world, 'termination_rules', []):
        condition(asdict(rule))
    for beat in getattr(world, 'beat_specs', []):
        for item in beat.completion_conditions:
            if isinstance(item, dict):
                condition(item)
    # A director override is still a real commitment even without a writer
    # declared at generation. It does not give actors new access to its value.
    for message in state.history:
        if message.authoritative:
            for op in message.state_patch:
                effect(op)
    return keys


def presented_entity(state, entity_id, spec):
    tracked = tracked_keys(state, entity_id)
    values = state.entity_states.get(entity_id, {})
    result = {'kind': spec.get('kind'),
              'state': {key: value for key, value in values.items() if key in tracked}}
    initial = {key: value for key, value in (spec.get('state') or {}).items() if key not in tracked}
    if initial:
        result['untracked_opening_state'] = initial
        result['tracking_note'] = ('这些键没有执行更新或条件依赖，只描述开场，不能据此否定后来已发生的普通动作；'
                                   '没有自动从文字核验当前值，不靠重复整理把它们假装更新。')
    return result
