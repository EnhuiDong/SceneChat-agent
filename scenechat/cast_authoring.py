"""Optional authored portrait then protocol binding; no runtime cast mutation."""
from dataclasses import asdict
import json

from .build_control import CURRENT_BUILD
from .prompt_library import prompt_contract
from .recovery import fingerprint
from .scenario import ScenarioValidationError


def draft_portraits(user_prompt, brief, world, invoke):
    sources = {'user': user_prompt, 'brief': brief.to_dict()}
    contract = prompt_contract('cast_portrait')
    input_key = fingerprint([sources, contract])
    if world is not None:
        sources.update({'opening': world.opening_scene,
                       'public_world': world.public_world_markdown, 'director_notes': world.director_notes_markdown,
                       'scoped_facts': [asdict(fact) for fact in world.facts], 'fixed_canon': world.fixed_canon})
    key = fingerprint([sources, contract])

    def validate(payload):
        rows = payload.get('portraits') if isinstance(payload, dict) else None
        if not isinstance(rows, list) or len(rows) != brief.requested_character_count:
            raise ScenarioValidationError(['人物稿 portraits 数量必须与原始约束人数一致'])
        names = []
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get('name'), str) or not row['name'].strip()
                    or not isinstance(row.get('portrait'), str) or not row['portrait'].strip()
                    or not isinstance(row.get('desires'), list) or not 1 <= len(row['desires']) <= 2
                    or any(not isinstance(value, str) or not value.strip() for value in row['desires'])):
                raise ScenarioValidationError(['每份人物稿要求 name、非空 portrait 和一至两项 desires；不填写接口或未来台词'])
            names.append(row['name'])
            details = row.get('known_details', [])
            if (not isinstance(details, list) or len(details) > 6 or
                    any(not isinstance(value, str) or not 1 <= len(value.strip()) <= 500 for value in details)):
                raise ScenarioValidationError(['known_details 只允许最多六条本人确知的简短资料；未知不补答案'])
        if len(set(names)) != len(names):
            raise ScenarioValidationError(['人物稿姓名不能重复'])

    control = CURRENT_BUILD.get()
    saved = control.recovery.get('cast_portrait') if control else None
    if isinstance(saved, dict) and (saved.get('key') == key or
            saved.get('preworld') is True and saved.get('input_key') == input_key):
        validate(saved.get('payload', {}))
        return saved['payload']
    # Keep inferred brief suggestions in the cache identity, not the author's
    # creative canon. The complete original user input remains authoritative.
    author_sources = {key:value for key,value in sources.items() if key != 'brief'}
    author_sources['character_count'] = brief.requested_character_count
    if brief.contradictions:
        author_sources['constraint_conflicts'] = brief.contradictions
    payload = invoke(contract, json.dumps(author_sources, ensure_ascii=False),
                     temperature=0.8, max_tokens=6000, validate_payload=validate)
    if control:
        control.recovery['cast_portrait'] = {'key': key, 'payload': payload, 'input_key':input_key, 'preworld':world is None}
        control.save_recovery()
    return payload


def bind_known_details(characters, portrait):
    """Literal owner-specific compile, so binding cannot drop authored facts."""
    by_name = {row['name']: row for row in portrait['portraits']}
    for actor in characters:
        details = by_name.get(actor.name, {}).get('known_details', [])
        if details:
            actor.knowledge['开场确知的具体资料'] = '\n'.join('- ' + value for value in details)
    return characters


def bind_portrait_fields(characters, portrait):
    """Compile authored wishes literally, not a second inferred task list."""
    by_name = {row['name']: row for row in portrait['portraits']}
    if set(by_name) != {actor.name for actor in characters} or len(characters) != len(by_name):
        raise ScenarioValidationError(['绑定角色姓名必须与已建立人物稿一一对应，不重命名、不增减人物'])
    for actor in characters:
        actor.goals = list(by_name[actor.name]['desires'])
        # Preserve the person's authored life and reasons for choice; protocol
        # binding must not turn a tendency into a new unconditional script.
        actor.decision_logic = by_name[actor.name]['portrait']
    return bind_known_details(characters, portrait)


BINDING_INSTRUCTION = '''本步只把已写好的人物稿绑定到 JSON 与既有运行接口，不做第二次人物创作。
完整保留与原始用户及世界一致的性格、具体处境、本人事实和愿望，分别映射到现有字段。
desires 按原文和原顺序映射为 goals，不添加整理工作等新目标，不降成保持自然、避免尴尬或物品工序；decision_logic 写取舍代价，不添未来问答程序。
不得给别的人物复制私有经历、未来知情或对方心理；不把已写人物压成职业加形容词。
人物姓名与原始设定一致；role 精确绑定已有标签，身份的自然描述另放 public_identity/public_background。
稿件只是自动补全候选，冲突时原始用户明确设定与世界优先；不改人数、机制或用户约束。'''
