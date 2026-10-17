"""Actor-owned, quote-grounded changes of outlook, not objective goal updates.

Receipts live only on accepted intents, so export/checkpoint/branch history is
the source of truth. No shared mind, new retrieval service or extra model call.
"""
import json

from .config import config_value

STANCES = {'pursuing', 'answered', 'revised', 'released'}
LABELS = {'pursuing': '仍在争取', 'answered': '已经得到回应',
          'revised': '本人已调整愿望', 'released': '本人暂时放下'}


def enabled():
    return config_value('simulation', 'interaction_memory', 'off') == 'evidence'


def validated_updates(state, actor, updates, *, before_turn=None):
    """Validate source/visibility/quotation, not the truth of an interpretation."""
    if not isinstance(updates, list):
        return []
    sources = {m.event_id: m for m in state.history}
    accepted = []
    for item in updates[:2]:
        if not isinstance(item, dict):
            continue
        source_id, quote, takeaway = (item.get(k) for k in ('source_event_id', 'quote', 'takeaway'))
        if not all(isinstance(value, str) for value in (source_id, quote, takeaway)):
            continue
        source = sources.get(source_id)
        if (source is None or source.intent.get('generation_fallback')
                or before_turn is not None and source.turn >= before_turn
                or not state._agent_can_observe(actor, source)):
            continue
        visible = ([source.individual_observations[actor.name]]
                   if actor.name in source.individual_observations else [source.action, source.speech])
        if not 2 <= len(quote) <= 240 or not any(isinstance(text, str) and quote in text for text in visible):
            continue
        takeaway = takeaway.strip()
        if not 1 <= len(takeaway) <= 240:
            continue
        goal, stance = item.get('goal', ''), item.get('stance', '')
        if not isinstance(goal, str) or not isinstance(stance, str):
            continue
        if goal and (goal not in actor.goals or stance not in STANCES):
            continue
        if not goal and stance:
            continue
        value = {'source_event_id': source_id, 'quote': quote, 'takeaway': takeaway,
                 'goal': goal, 'stance': stance}
        if value not in accepted:
            accepted.append(value)
    return accepted


def receipts(state, actor):
    """Replay latest private outlook per opening wish or encountered speaker."""
    if not enabled():
        return []
    by_id = {m.event_id: m for m in state.history}
    latest = {}
    for message in state.history:
        if (message.speaker != actor.name or not message.authoritative
                or message.intent.get('generation_fallback')
                or not state._agent_can_observe(actor, message)):
            continue
        for item in validated_updates(state, actor, message.intent.get('interaction_updates'), before_turn=message.turn):
            source = by_id[item['source_event_id']]
            key = ('goal', item['goal']) if item['goal'] else ('speaker', source.speaker)
            latest[key] = {**item, 'source_speaker': source.speaker, 'recorded_at_turn': message.turn}
    return sorted(latest.values(), key=lambda item: item['recorded_at_turn'])[-8:]


def render(state, actor):
    rows = receipts(state, actor)
    if not rows:
        return ''
    lines = ['【交流已经给自己留下的变化——本人看法，不是对方内心或世界结算】',
             '开场愿望仍是人物来由，下面是你后来形成的取舍；已得到回应不等于愿望实现，暂时放下也不抹除原设定。'
             '新事件可以改变这些看法；引用可能是对方的谎话，不能当作核验。']
    for row in rows:
        lines.append(json.dumps({'source_event_id': row['source_event_id'], '当时实际可见的原话或动作': row['quote'],
            '来源人物': row['source_speaker'], '你后来留下的看法': row['takeaway'],
            **({'开场愿望': row['goal'], '本人当时的取舍': LABELS[row['stance']]} if row['goal'] else {})}, ensure_ascii=False))
    return '\n'.join(lines)


ANNOTATION_INSTRUCTION = '''可选私有 interaction_updates，最多两条；仅当已发生的交流确实改变你此刻的取舍时记录，不为填表编变化。
格式 [{"source_event_id":"本人已可见的既有事件ID","quote":"该事件speech或action中逐字原文2至240字",
"takeaway":"这件事给自己留下的具体看法或当前愿望，不分析对方未公开的心理"}]。
若涉及自己的某项开场目标，可加 goal=本人goals中的完整原文、stance=pursuing|answered|revised|released。
它们分别只表示自己仍争取、已收到回应、调整愿望、暂时放下；不改变引擎goal_status、不证明对方承诺已兑现。
不引用本次尚未提交的行为，不预写下轮台词，不记录同一看法的换词复述。无变化就省略。'''
