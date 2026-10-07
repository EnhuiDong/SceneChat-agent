"""Extractive, character-owned memory: compression never promotes claims to facts."""
from dataclasses import asdict
import re


def archive_and_trim(agent):
    events = [m for m in agent.memories if m.memory_type == "event"]
    structured = [m for m in agent.memories if m.memory_type != "event"]
    # Unresolved commitments and causal evidence have no age-based expiry.
    protected = [m for m in structured if m.active]
    recent = events[-30:] + [m for m in structured if not m.active][-20:]
    kept = {id(m) for m in protected + recent}
    existing = {(m.get("event_id"), m.get("content")) for m in agent.memory_archive}
    for m in agent.memories:
        if id(m) not in kept and (m.event_id, m.content) not in existing:
            agent.memory_archive.append(asdict(m))
    agent.memories = sorted(protected + recent, key=lambda m: m.created_at_turn)


def recall(agent, focus_agents=(), limit=14, *, represented_observations=(), excluded_event_ids=()):
    terms = set(re.findall(r"[\w]{2,}", " ".join(agent.goals + [agent.current_location, agent.current_conversation_goal])))
    for term in list(terms):
        if re.search(r"[\u4e00-\u9fff]", term):
            terms.update(term[i:i + 2] for i in range(len(term) - 1))
    focus = set(focus_agents)
    records = [asdict(m) for m in agent.memories] + agent.memory_archive
    represented = set(represented_observations)
    excluded = set(excluded_event_ids)
    unique = {}
    for m in records:
        if m.get("event_id") in excluded:
            continue
        scopes = m.get("visibility") or []
        if not any(s in {"public", "agent_private", f"agent:{agent.name}"} for s in scopes):
            continue
        if (m.get("memory_type") == "event" and m.get("source") == "direct_observation"
                and (m.get("event_id"), m.get("content")) in represented):
            continue
        unique.setdefault((m.get("event_id"), m.get("content")), m)
    def rank(m):
        return (m.get("active", True) and m.get("memory_type") == "commitment",
                bool(focus.intersection(m.get("related_agents", []))),
                sum(t in m.get("content", "") for t in terms),
                m.get("importance", 1), m.get("created_at_turn", 0))
    chosen = sorted(unique.values(), key=rank, reverse=True)[:limit]
    lines = [
        f"- [event_id={m.get('event_id')}|{m.get('memory_type')}|"
        f"{'未解决/有效' if m.get('active', True) else '已归档/失效'}|"
        f"来源={m.get('source')}；仅为当时观察或主张，不代表已验证] {m.get('content')}"
        for m in chosen
    ]
    # Omitting a raw duplicate must not omit later corrections of that claim.
    selected_events = {m.get("event_id") for m in chosen} | {event_id for event_id, _ in represented}
    relevant_beliefs = [b for b in agent.belief_records if b.source_event_id in selected_events
                       and b.source_event_id not in excluded]
    included = {b.id for b in relevant_beliefs}
    # Follow corrections even when the correction itself is no longer recent.
    for belief in agent.belief_records:
        if (belief.supersedes in included and belief.id not in included
                and belief.source_event_id not in excluded):
            relevant_beliefs.append(belief)
            included.add(belief.id)
    lines = [f"- [认知记录 {b.id}|{b.epistemic_status}|{'有效' if b.active else '已失效'}|"
             f"event_id={b.source_event_id}|修正={b.supersedes or '无'}] {b.content}"
             for b in relevant_beliefs] + lines
    eligible_summaries = [summary for summary in agent.memory_summaries
                          if not excluded.intersection(summary.get("source_event_ids", []))]
    if eligible_summaries:
        summary = eligible_summaries[-1]
        lines.append(f"- 阶段摘录（不构成新的已验证事实；来源 {','.join(summary.get('source_event_ids', []))}）：{summary.get('content', '')}")
    return "\n".join(lines) or "- 暂无故事记忆"
