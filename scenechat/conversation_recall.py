"""Bounded, actor-visible dialogue receipts; not summaries or verified facts."""
from collections import Counter
import math
import re

from .config import config_value


def _terms(text):
    # Lexical fallback supports both spaced languages and Chinese. It retrieves
    # quotations, not semantic judgements; no genre-specific vocabulary.
    terms = set(re.findall(r"[a-zA-Z0-9_]{2,}", text.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(run[i:i + 2] for i in range(len(run) - 1))
    return terms


def recall_events(state, agent, recent, focus=None):
    if config_value("simulation", "conversation_recall", "off") != "linked":
        return []
    visible = [m for m in state.history[-160:]
               if m.kind == "dialogue" and state._agent_can_observe(agent, m)
               and not m.intent.get("generation_fallback")]
    by_id = {m.event_id: m for m in visible}
    presented = {m.event_id for m in recent}
    anchors = list(recent[-2:])
    if focus is not None:
        anchors.append(focus)
    own = next((m for m in reversed(visible) if m.speaker == agent.name), None)
    if own is not None:
        anchors.append(own)
    anchors = list({m.event_id: m for m in anchors}.values())
    # An individual observation replaces the event. Never use the hidden
    # original Intent's reply graph, addressees or thread to rank its content.
    linked = {}
    for anchor in anchors:
        current = anchor
        for depth in range(3):
            if agent.name in current.individual_observations:
                break
            parent = by_id.get(str(current.intent.get("reply_to_event_id") or ""))
            if parent is None or parent.turn >= current.turn:
                break
            linked[parent.event_id] = max(linked.get(parent.event_id, 0), 12 - depth * 3)
            current = parent
    documents = {m.event_id: _terms(m.as_observation_for(agent.name)) for m in visible}
    frequency = Counter(term for terms in documents.values() for term in terms)
    query = set().union(*(_terms(m.as_observation_for(agent.name)) for m in anchors)) if anchors else set()
    ranked = []
    for message in visible:
        if message.event_id in presented:
            continue
        shared = documents[message.event_id] & query
        lexical = (sum(math.log(1 + len(visible) / frequency[t]) for t in shared)
                   / math.sqrt(max(len(documents[message.event_id]), 1))) if len(shared) >= 3 else 0
        link = linked.get(message.event_id, 0)
        if agent.name not in message.individual_observations:
            parent_id = str(message.intent.get("reply_to_event_id") or "")
            if parent_id in linked:
                link = max(link, 5)
        score = link + lexical
        if score >= 1.2:
            ranked.append((score, message.turn, message))
    selected, speakers = [], Counter()
    for _, _, message in sorted(ranked, key=lambda row: (row[0], row[1]), reverse=True):
        if speakers[message.speaker] >= 2:
            continue
        selected.append(message)
        speakers[message.speaker] += 1
        if len(selected) == 6:
            break
    return sorted(selected, key=lambda m: m.turn)


def render_recall(state, agent, recent, focus=None):
    events = recall_events(state, agent, recent, focus)
    if not events:
        return ""
    lines = ["【较早交流回看——原话节选，不是已核验事实或新的待办】",
             "这些话在近期窗口之前发生。已经有人回答不等于答案可信，但不能当成从未回答；"
             "态度变化也不自动证明身份、罪责或前后矛盾。保留分歧，是否改口由本人动机决定。"
             "记录可能不全；不要从缺少记录推断此事没发生。"]
    for message in events:
        observation = message.as_observation_for(agent.name)
        suffix = "…（节选）" if len(observation) > 500 else ""
        lines.append(f"- [{message.event_id}] {observation[:500]}{suffix}")
    return "\n".join(lines)
