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
    return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def _similar(left: str, right: str, threshold: float) -> bool:
    first = _normalized(left)
    second = _normalized(right)
    if min(len(first), len(second)) < 12:
        return False
    return SequenceMatcher(None, first, second).ratio() >= threshold


def _known_text(agent: AgentState) -> str:
    return _normalized("\n".join(
        [agent.profile, *agent.observations, *agent.private_memory]
        + list(agent.known_facts.values())
    ))


def _hidden_fragments(state: SimulationState, agent: AgentState) -> list[str]:
    known = _known_text(agent)
    fragments: list[str] = []
    for fact_id, fact in state.facts.items():
        content = str(getattr(fact, "content", "") or "").strip()
        normalized = _normalized(content)
        if fact_id not in agent.known_facts and len(normalized) >= 8 and normalized not in known:
            fragments.append(content)

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
                or normalized in known
                or line in {"无额外秘密", "未提供额外私有知识。"}
            ):
                continue
            fragments.append(line)
    return fragments


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
        ))

    private_output = "\n".join([
        str(getattr(intent, "action", "") or ""),
        speech,
        *[
            str(item.get("content") or "")
            for item in getattr(intent, "memory_candidates", [])
            if isinstance(item, dict)
        ],
    ])
    normalized_output = _normalized(private_output)
    leaked = next((
        fragment for fragment in _hidden_fragments(state, agent)
        if _normalized(fragment) in normalized_output
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
