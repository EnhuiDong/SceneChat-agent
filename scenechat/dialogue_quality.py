from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .models import AgentState, SimulationState


@dataclass(frozen=True)
class DialogueQualityIssue:
    code: str
    message: str
    hard: bool = False


def _normalized(value: Any) -> str:
    text = str(value or "").lower()
    replacements = {
        "害怕": "担心", "担忧": "担心", "恐怕": "担心",
        "丢掉": "失去", "保不住": "失去", "饭碗": "职位", "工作": "职位",
        "同伙": "队友", "同伴": "队友", "一伙": "队友",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _similar(left: str, right: str, threshold: float) -> bool:
    first = _normalized(left)
    second = _normalized(right)
    if min(len(first), len(second)) < 12:
        return False
    return SequenceMatcher(None, first, second).ratio() >= threshold


def _looks_like_secret(fragment: str, output: str) -> bool:
    secret = _normalized(fragment)
    candidate = _normalized(output)
    if min(len(secret), len(candidate)) < 8:
        return False
    if secret in candidate:
        return True
    if SequenceMatcher(None, secret, candidate).ratio() >= 0.68:
        return True
    secret_pairs = {secret[index:index + 2] for index in range(len(secret) - 1)}
    output_pairs = {candidate[index:index + 2] for index in range(len(candidate) - 1)}
    shared = secret_pairs.intersection(output_pairs)
    if len(shared) >= 5 and len(shared) / max(len(secret_pairs), 1) >= 0.58:
        return True
    secret_characters = set(secret)
    shared_characters = secret_characters.intersection(set(candidate))
    return (
        len(shared_characters) >= 6
        and len(shared_characters) / max(len(secret_characters), 1) >= 0.68
    )


def _known_parts(agent: AgentState) -> list[str]:
    profile_lines = [
        line for line in agent.profile.splitlines()
        if "不知道" not in line and "未知" not in line
    ]
    return [
        part for part in (
            [*profile_lines, *agent.observations, *agent.private_memory]
        + list(agent.known_facts.values())
        + [item.content for item in agent.belief_records if item.active]
        ) if str(part).strip()
    ]


def _secret_units(value: str) -> list[str]:
    text = str(value or "").strip()
    units = [text, *re.split(r"[。！？!?；;\n]+", text)]
    result = []
    for unit in units:
        candidate = unit.strip().lstrip("-* ").strip()
        if len(_normalized(candidate)) >= 8 and candidate not in result:
            result.append(candidate)
    return result


def _hidden_fragments(state: SimulationState, agent: AgentState) -> list[str]:
    known_parts = _known_parts(agent)
    normalized_known = _normalized("\n".join(known_parts))
    fragments: list[str] = []
    for fact_id, fact in state.facts.items():
        content = str(getattr(fact, "content", "") or "").strip()
        normalized = _normalized(content)
        if fact_id in agent.known_facts:
            continue
        if (
            len(normalized) >= 8
            and normalized not in normalized_known
            and not any(_looks_like_secret(content, part) for part in known_parts)
        ):
            fragments.extend(_secret_units(content))
        for proposition in getattr(fact, "protected_propositions", []) or []:
            value = str(proposition).strip()
            proposition_normalized = _normalized(value)
            if (
                len(proposition_normalized) >= 8
                and proposition_normalized not in normalized_known
                and not any(_looks_like_secret(value, part) for part in known_parts)
            ):
                fragments.extend(_secret_units(value))

    for other in state.agents.values():
        if other.name == agent.name:
            continue
        public = _normalized(other.public_profile)
        for raw_line in other.profile.splitlines():
            line = raw_line.strip().lstrip("-* ").strip()
            normalized = _normalized(line)
            if (
                not line
                or line.startswith("#")
                or len(normalized) < 8
                or normalized in public
                or normalized in normalized_known
                or any(_looks_like_secret(line, part) for part in known_parts)
                or line in {"无额外秘密", "未提供额外私有知识。"}
            ):
                continue
            fragments.extend(_secret_units(line))
    return list(dict.fromkeys(fragments))


def inspect_dialogue_intent(
    state: SimulationState,
    agent: AgentState,
    intent: Any,
) -> list[DialogueQualityIssue]:
    """Check dialogue quality without granting the model any new information."""

    issues: list[DialogueQualityIssue] = []
    speech = str(getattr(intent, "speech", "") or "").strip()
    if speech:
        own_recent = [
            message.speech for message in state.history
            if message.speaker == agent.name and message.speech.strip()
        ][-3:]
        normalized_speech = _normalized(speech)
        if any(
            _similar(speech, previous, 0.88)
            or (
                len(_normalized(previous)) >= 12
                and _normalized(previous) in normalized_speech
            )
            for previous in own_recent
        ):
            issues.append(DialogueQualityIssue(
                "self_repetition",
                "台词与该角色最近的表达过于相似；保留意图，但换一种具体回应方式。",
            ))

        previous_visible = next((
            message for message in reversed(state.history)
            if message.speaker != agent.name
            and message.speech.strip()
            and state._agent_can_observe(agent, message)
        ), None)
        if previous_visible and _similar(speech, previous_visible.speech, 0.84):
            issues.append(DialogueQualityIssue(
                "parroting",
                f"台词近似复述了{previous_visible.speaker}刚才的话；请给出新的反应、判断或行动。",
            ))

        sentence_count = len([part for part in re.split(r"[。！？!?；;]+", speech) if part.strip()])
        if len(speech) > 240 or sentence_count > 4:
            issues.append(DialogueQualityIssue(
                "monologue",
                "台词像一次完整演讲；压缩到一至三句，只推进一个主要意图。",
            ))

    pending_ids = {
        str(item.get("event_id") or "") for item in agent.pending_intents
        if str(item.get("event_id") or "")
    }
    reply_to = str(getattr(intent, "reply_to_event_id", "") or "")
    if pending_ids and reply_to not in pending_ids:
        most_recent = agent.pending_intents[-1]
        issues.append(DialogueQualityIssue(
            "missing_response",
            f"本轮没有回应{most_recent.get('speaker')}提出的{most_recent.get('move')}；"
            f"请填写 event_id={most_recent.get('event_id')}，即使选择回避或沉默也要明确回应。",
            hard=True,
        ))

    private_output = "\n".join([
        str(getattr(intent, "action", "") or ""),
        speech,
        *[
            str(item.get("content") or "")
            for item in getattr(intent, "memory_candidates", [])
            if isinstance(item, dict)
        ],
        *[
            str(item.get("content") or "")
            for item in getattr(intent, "claim_updates", [])
            if isinstance(item, dict)
        ],
        *[
            str(item.get("private_note") or "")
            for item in getattr(intent, "relationship_updates", {}).values()
            if isinstance(item, dict)
        ],
    ])
    leaked = next((
        fragment for fragment in _hidden_fragments(state, agent)
        if _looks_like_secret(fragment, private_output)
    ), "")
    if leaked:
        issues.append(DialogueQualityIssue(
            "secret_leak",
            "输出包含当前角色尚未获知的私密事实；删除该信息，只依据已观察内容行动。",
            hard=True,
        ))
    return issues


def quality_retry_instruction(issues: list[DialogueQualityIssue]) -> str:
    return "\n".join(f"- [{issue.code}] {issue.message}" for issue in issues)


def inspect_narration_event(
    state: SimulationState,
    narration: str,
    *,
    visibility: str,
) -> list[DialogueQualityIssue]:
    """Reject narrator loops and impossible closed-cast references."""

    text = str(narration or "").strip()
    issues: list[DialogueQualityIssue] = []
    recent = [
        message.speech for message in state.history[-12:]
        if message.kind == "narration" and message.speech.strip()
    ]
    if any(
        _similar(text, previous, 0.82)
        or (
            len(_normalized(previous)) >= 8
            and (
                _normalized(previous) in _normalized(text)
                or _normalized(text) in _normalized(previous)
            )
        )
        for previous in recent[-6:]
    ):
        issues.append(DialogueQualityIssue(
            "narration_repetition",
            "旁白与近期事件重复；不要再次描述同一灯光、倒计时、广播或人物小动作，必须带来新的可观察变化。",
            hard=True,
        ))

    # The runtime cannot create new actors.  Names introduced specifically as
    # the next speaker/selected participant are therefore always invalid,
    # while ordinary place names and background NPC prose remain untouched.
    ignored = {"所有人", "每个人", "下一位", "参与者", "玩家", "众人"}
    selected_names = re.findall(
        r"(?:点名(?:者)?|下一位(?:发言者|玩家|参与者)|轮到)\s*[:：]?\s*"
        r"([A-Za-z][A-Za-z0-9_.-]{1,39}|[\u4e00-\u9fff]{2,4})",
        text,
    )
    unknown = [
        name for name in selected_names
        if name not in state.agents and name not in ignored
    ]
    if unknown:
        issues.append(DialogueQualityIssue(
            "unknown_cast_reference",
            f"旁白把不存在的角色“{unknown[0]}”当作行动者；只能点名当前角色名单中的人物。",
            hard=True,
        ))

    if visibility == "audience_only" and not state.ended:
        explicit_reveals = []
        compact = _normalized(text)
        for agent in state.agents.values():
            faction = _normalized(agent.faction)
            if not faction:
                continue
            for template in (
                f"{agent.name}是{agent.faction}",
                f"{agent.name}真实身份是{agent.faction}",
                f"{agent.name}属于{agent.faction}",
            ):
                if _normalized(template) in compact:
                    explicit_reveals.append(agent.name)
                    break
        if explicit_reveals:
            issues.append(DialogueQualityIssue(
                "premature_identity_reveal",
                "读者镜头在公开揭晓或结算前直接确认了隐藏身份；改成可多重解释的线索。",
                hard=True,
            ))
    return issues
