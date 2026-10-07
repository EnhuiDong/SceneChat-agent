"""Source-backed annotations of ordinary actions, never new engine facts.

The journal is derived from persisted message intents. It has no separate store,
does not parse prose into state patches, and never grants knowledge or authority.
"""
from .config import config_value


def validated_notes(notes, action):
    if not isinstance(notes, list) or not isinstance(action, str):
        return []
    accepted = []
    for item in notes[:3]:
        if not isinstance(item, dict):
            continue
        subject, quote = item.get("subject"), item.get("quote")
        if not isinstance(subject, str) or not isinstance(quote, str):
            continue
        subject, quote = subject.strip(), quote.strip()
        if (not 1 <= len(subject) <= 40 or not 6 <= len(quote) <= 400
                or any(c in subject for c in "\n\r[]{}")
                or subject not in quote or quote not in action):
            continue
        entry = {"subject": subject, "quote": quote}
        if entry not in accepted:
            accepted.append(entry)
    return accepted


def enabled():
    return config_value("simulation", "scene_continuity_mode", "off") == "evidence_log"


def journal(state, *, agent=None, public_only=True, max_subjects=18):
    """Latest *visible* annotation per exact label; aliases are not inferred."""
    if not enabled():
        return ""
    latest = {}
    for message in state.history:
        if (message.speaker not in state.agents or not message.authoritative
                or message.intent.get("generation_fallback")):
            continue
        if agent is not None:
            if not state._agent_can_observe(agent, message) or agent.name in message.individual_observations:
                continue
        elif public_only and (message.visibility != "public" or "public" not in message.scopes):
            continue
        for note in validated_notes(message.intent.get("continuity_notes"), message.action):
            latest[note["subject"]] = (message.turn, message.event_id, message.speaker, note["quote"])
    rows = sorted(latest.items(), key=lambda item: item[1][0])[-max_subjects:]
    if not rows:
        return ""
    import json
    return ("【普通行动连续性记录——事件原文，不是新的规则结果】\n"
            "只保留同标签最新可见记录，标签别名不自动合并；没有记录不表示没有变化。"
            "动作原文表示该人物已提交的行为，不证明检查成功、他人配合或外部已回应。"
            "较新的事件与权威状态优先，不能用旧开场把已移动物品恢复原位。\n"
            + "\n".join(f"- {json.dumps(subject, ensure_ascii=False)} [{event_id}] 第{turn}事件 {speaker}："
                        + json.dumps(quote, ensure_ascii=False)
                        for subject, (turn, event_id, speaker, quote) in rows))


ANNOTATION_INSTRUCTION = """可选 continuity_notes=[{"subject":"本次改变位置或使用状态的具体对象原名",
"quote":"从本次 action 逐字复制的完整相关动作"}]，最多三条。没有变化就省略。
subject 必须出现在 quote，quote 必须是 action 的原文，不能来自台词、愿望或背景。
它只帮助下轮记住你已经做的普通动作，不修改客观世界，不替检查、投票或规则结算宣告结果。
不要为了填记录增加动作或复述上一轮。"""
