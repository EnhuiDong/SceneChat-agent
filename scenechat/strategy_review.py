"""Opt-in matched strategy probes and human-readable, blinded review.

No production mode is changed and no model is called by this module. Reports
contain story data and must be saved privately, not committed to a public repo.
"""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path

from .chat_prompt import ChatPrompt


STRATEGIES = {
    'production': {'perspective': 'current', 'bias': 'current'},
    'identity': {'perspective': 'identity', 'bias': False},
    'director': {'perspective': 'director', 'bias': False},
    'identity_bias': {'perspective': 'identity', 'bias': True},
    'director_bias': {'perspective': 'director', 'bias': True},
}


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def strategy_prompt(original, actor_name, strategy, opportunity=''):
    """Change framing/opportunity only; retain the exact knowledge and schema.

    'Director' is an external writing stance with the SAME scoped information,
    not the actual omniscient director agent. No historical turns are invented.
    """
    if strategy not in STRATEGIES:
        raise ValueError('Unknown comparison strategy')
    if strategy == 'production':
        return original
    if len(getattr(original, 'messages', ())) != 4:
        raise ValueError('Matched strategy comparison requires four original messages')
    messages = list(original.messages)
    role, exchange = messages[-1]
    if opportunity:
        marker = '\n\n' + opportunity
        if marker not in exchange:
            raise ValueError('Opportunity text does not match the original input')
        exchange = exchange.replace(marker, '', 1)
    identity = STRATEGIES[strategy]['perspective'] == 'identity'
    frame = (
        f'你就是{actor_name}。这些是你的经历、所知和眼前的事情。此刻不是从外面分析这个人应该怎样演，'
        '而是从自己的位置回应：你在意什么，眼前这句话让你怎样，你现在愿意说什么、做什么。'
        '可以不确定、笨拙、理性或冲动，由处境和你自身决定；不为剧情好看替全场安排选择。'
        if identity else
        f'你是编写{actor_name}本次言行的导演。只使用此人同样有限的所知，'
        '从外部分析其性格、关系与当前处境，拟写符合此人设定的一次回应。'
        '不以自己身份回应，不写其他人物的新决定，不用隐藏真相或未来情节，'
        '不输出导演评语或分析，只提交该人的公开言行和同样的执行字段。'
    )
    role0, system = messages[0]
    _, _, remaining = system.partition('\n')
    # Retain the writing contract verbatim; the stance is the experimental
    # instruction, not a simultaneous deletion of rules or personality.
    messages[0] = (role0, frame + '\n' + remaining + '\n【此次生成立场】' + frame)
    if identity:
        exchange = exchange.replace(
            '这个人的性格、关系与此刻牵挂，会让他怎样接这句话、面对这件事？',
            '你的性格、关系与此刻牵挂，会让你怎样接这句话、面对这件事？')
        exchange = exchange.replace('选择他现在真愿意说或做的一次反应', '选择你现在真愿意说或做的一次反应')
        exchange = exchange.replace('这个人此刻的新计划与实际感受', '你此刻的新计划与实际感受')
    exchange += '\n【本次立场】' + frame
    if STRATEGIES[strategy]['bias'] and opportunity:
        exchange += '\n\n' + opportunity
    messages[-1] = (role, exchange)
    return ChatPrompt(messages)


def validate_comparison(rows):
    """Reject a claimed matched comparison when its initial inputs differ."""
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row['case'], row['model'], row.get('trial', 1), row['variant'])
        if identity in seen:
            raise ValueError('Repeated comparison cell; retain a distinct trial number')
        seen.add(identity)
        if row['variant'] not in STRATEGIES:
            raise ValueError('Unknown strategy in results')
        groups[identity[:3]].append(row)
    for group in groups.values():
        for key in ('initial_state_hash', 'scenario_hash', 'reference_hash', 'protocol_hash'):
            values = {row.get(key) for row in group}
            if None in values or len(values) != 1:
                raise ValueError(f'Matched comparison has different or missing {key}')
    return groups


def review_markdown(rows, *, title='同场景策略对照', reveal=False):
    groups = validate_comparison(rows)
    lines = [f'# {title}', '',
             '先阅读匿名原文，再看评价和策略答案。信息、设定与初始状态相同；'
             '后续分支由各自实际选择产生。单次采样不是稳定优势证明。', '',
             '原始输出、请求和人物秘密留在本地 JSON；下文不展示模型的私密理由、记忆或分析。', '']
    answer_key = []
    for (case, model, trial), group in sorted(groups.items()):
        # Deterministic independent presentation order, not quality sorting.
        ordered = sorted(group, key=lambda row: fingerprint([case, model, trial, row['variant']]))
        lines.extend([f'## {case} / {model} / 样本 {trial}', ''])
        if group[0].get('context'):
            lines.extend([group[0]['context'], ''])
        for index, row in enumerate(ordered):
            label = chr(ord('A') + index)
            lines.extend([f'### 版本 {label}' + (f"（{row['variant']}）" if reveal else ''), ''])
            events = row.get('events') or []
            if events:
                for event in events:
                    turn = f"（事件 {event['turn']}）" if event.get('turn') is not None else ''
                    lines.append(f"{event.get('speaker', '人物')}{turn}：")
                    if event.get('visibility') == 'audience_only':
                        lines.append('（仅读者可见，不进入角色所知）')
                    if event.get('action'):
                        lines.append('动作：' + event['action'])
                    lines.append('对白：' + (event.get('speech') or '（未说话）'))
                    lines.append('')
            else:
                lines.extend(['本格未产生可展示的有效言行；原始输出仍保留。', ''])
            status = row.get('status', 'not_committed')
            lines.extend([f"执行状态：{status}；已知 tokens：{row.get('tokens') if row.get('tokens') is not None else '未知'}。", ''])
            summary = row.get('summary')
            if summary:
                lines.extend([f"新增事件：{summary.get('new_events', '未知')}；兜底：{summary.get('fallbacks', '未知')}；"
                              f"实际观察到故事结束：{'是' if summary.get('ending_observed') else '否'}。", ''])
            if row.get('shared_sample_from'):
                lines.extend(['该格输入与另一格完全相同，复用同一输出，未重复调用模型；不构成独立样本。', ''])
            if row.get('opportunity_active') is False:
                lines.extend(['本节点未触发推进机会，不能用这一格判断推进因子的效果。', ''])
            answer_key.append(f'- {case}/{model}/样本{trial}/{label}：{row["variant"]}')
    lines.extend(['## 策略答案（建议先完成自己的判断）', '', *answer_key, '',
                  'AI评价需要引用具体原文，并区分相对更好与绝对达标；不因台词长、'
                  '结束快、字段合法或情绪强烈自动判优。未观察到结局时不评价完整收尾。', ''])
    return '\n'.join(lines)


def main():
    """Offline rendering only; never invokes a provider or runs a simulation."""
    parser = argparse.ArgumentParser(description='Render retained matched strategy rows for private human review')
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reveal', action='store_true')
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.results.resolve():
        parser.error('Review output must not overwrite the source results')
    rows = json.loads(args.results.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        parser.error('Results must be an array of retained comparison rows')
    report = review_markdown(rows, reveal=args.reveal)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding='utf-8')
    else:
        print(report)


if __name__ == '__main__':
    main()
