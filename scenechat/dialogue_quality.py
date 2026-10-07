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


def overlong_dialogue(speech: str) -> bool:
    """A soft size warning, not a rule that humans may only say four sentences."""
    sentence_count = len([part for part in re.split(r"[。！？!?；;]+", speech) if part.strip()])
    return len(speech) > 480 or (len(speech) > 240 and sentence_count > 8)


def _normalized(value: Any) -> str:
    text = str(value or "").lower()
    replacements = {
        "没有": "没",  # Preserve negation while matching ordinary contracted wording.
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


def _reordered_narration(left: str, right: str) -> bool:
    first, second = _normalized(left), _normalized(right)
    if min(len(first), len(second)) < 20:
        return False
    first_pairs = {first[i:i + 2] for i in range(len(first) - 1)}
    second_pairs = {second[i:i + 2] for i in range(len(second) - 1)}
    shared = len(first_pairs & second_pairs)
    return shared >= 12 and 2 * shared / max(len(first_pairs) + len(second_pairs), 1) >= 0.68


def _looks_like_secret(fragment: str, output: str) -> bool:
    secret = _normalized(fragment)
    candidate = _normalized(output)
    if min(len(secret), len(candidate)) < 8:
        return False
    if secret in candidate:
        return True
    secret_pairs = {secret[index:index + 2] for index in range(len(secret) - 1)}
    def compact(value: str) -> str:
        return re.sub(r"自己|那次|那场|这个|的|会|让|了", "", value)
    # Compare coherent clauses, not a bag of characters from speech + notes +
    # memory. Common characters scattered over a long Intent are not a leak.
    for part in re.split(r"[。！？!?；;\n]+", str(output or "")):
        clause = _normalized(part)
        if len(clause) < 8:
            continue
        compact_secret, compact_clause = compact(secret), compact(clause)
        if (SequenceMatcher(None, secret, clause).ratio() >= 0.68
                or min(len(compact_secret), len(compact_clause)) >= 6
                and SequenceMatcher(None, compact_secret, compact_clause).ratio() >= 0.66):
            return True
        output_pairs = {clause[index:index + 2] for index in range(len(clause) - 1)}
        shared = secret_pairs.intersection(output_pairs)
        if (len(shared) >= 5 and len(shared) / max(len(secret_pairs), 1) >= 0.65
                and len(shared) / max(len(output_pairs), 1) >= 0.35):
            return True
    return False


def _private_profile_lines(agent: AgentState) -> list[str]:
    from .character_parser import _sections
    sections = _sections(agent.profile)
    if 5 not in sections:
        return agent.profile.splitlines()  # Legacy/free-form dossiers.
    # Stable personality and decision tendencies are not hidden evidence.
    # Hard privacy checks protect identities and private knowledge; arbitrary
    # motive secrecy can additionally use protected_propositions in world facts.
    identity = sections[5][1].splitlines()
    knowledge = re.split(r"[。；;\n]+", sections.get(8, ("", ""))[1])
    # The Markdown key is metadata, not part of the protected proposition.
    # "**确定知道**：<public fact>" otherwise fails the public-profile check
    # and turns an ordinary recollection into a false secret leak. Strip only
    # the generated Markdown field label, not arbitrary prose before a colon.
    return identity + [
        re.sub(r"^\s*[-*]?\s*\*\*[^*\n]+\*\*[：:]\s*", "", line)
        for line in knowledge
        if not any(marker in line for marker in ("不知道", "未知", "怀疑", "可能错误相信"))
    ]


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
        for raw_line in _private_profile_lines(other):
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
            or _reordered_narration(speech, previous)
            or (
                len(_normalized(previous)) >= 12
                and _normalized(previous) in normalized_speech
            )
            for previous in own_recent
        ):
            issues.append(DialogueQualityIssue(
                "self_repetition",
                "该判断已表达过；换词仍然是重复。给出新依据、回应具体问题或改用合法行动，不要重演同一观点。",
                hard=True,
            ))

        other_recent = [
            message for message in reversed(state.history)
            if message.speaker != agent.name
            and message.speaker in state.agents
            and message.speech.strip()
            and state._agent_can_observe(agent, message)
        ][:6]
        previous_visible = next((message for message in other_recent
                                 if _similar(speech, message.speech, 0.84)
                                 or _reordered_narration(speech, message.speech)), None)
        if previous_visible:
            issues.append(DialogueQualityIssue(
                "parroting",
                f"台词近似复述了{previous_visible.speaker}刚才的话；不要换姓名或场地复用同一段经历。直接回答当前问题、提出新依据或执行合法行动，简短同意即可，不再复述已有方案。",
                hard=len(normalized_speech) >= 24,
            ))

        if overlong_dialogue(speech):
            issues.append(DialogueQualityIssue(
                "monologue",
                "台词篇幅过长；保留当场必要的解释和回应，删去重复铺垫与总结，不强行把自然交流压成一句话。",
            ))

        if (getattr(intent, "action_type", "") == "vote"
                and len(normalized_speech) >= 25 and getattr(intent, "target", "")):
            same_target = [
                message for message in state.history
                if message.speaker != agent.name and message.speech.strip()
                and any(operation.get("op") == "record_vote"
                        and operation.get("target") == intent.target
                        and operation.get("actor") in state.votes
                        for operation in message.state_patch)
            ][-5:]
            if len(same_target) >= 2 and any(
                _similar(speech, message.speech, 0.67)
                or _reordered_narration(speech, message.speech)
                for message in same_target
            ):
                issues.append(DialogueQualityIssue(
                    "vote_rationale_echo",
                    "仍可投同一人，但理由几乎沿用前面选票。给出此人真正不同的依据或顾虑；"
                    "若只是跟票，可简短承认或只提交投票动作，不要伪装成独立推理。",
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

    # A stalled task needs a new decision, not the same request restated with
    # stronger wording. Only compare the owner's own task in the same thread;
    # a distinct target or genuinely different request remains allowed.
    move = str(getattr(intent, "conversation_move", "") or "")
    thread_id = str(getattr(intent, "thread_id", "") or "")
    addressed = set(getattr(intent, "addressed_to", []) or [])
    if move in {"question", "request", "challenge"} and speech:
        stalled = next((
            task for task in reversed(list(getattr(state, "agenda", {}).values()))
            if task.status == "blocked" and task.owner == agent.name
            and task.thread_id == thread_id
            and addressed.intersection(task.targets)
            and ((_normalized(speech) == _normalized(task.title)
                  and len(_normalized(speech)) >= 4)
                 or _similar(speech, task.title, 0.83))
        ), None)
        if stalled is not None:
            issues.append(DialogueQualityIssue(
                "stalled_task_repeat",
                "这项请求已有回应或拒绝；不要原样再问。请用新证据核验、改变具体条件、执行合法替代行动，或明确搁置。",
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


def sanitize_private_annotations(state: SimulationState, agent: AgentState, intent: Any) -> list[str]:
    """Drop suspect bookkeeping instead of discarding a safe performance.

    Disclosure matching is conservative and may reject a faithful paraphrase
    of a heard statement. Never whitelist a hidden fact or promote that
    paraphrase: retain the actual scoped dialogue, omit the suspect new note.
    Public action/speech still pass the unchanged hard privacy gate.
    """
    fragments = _hidden_fragments(state, agent)
    removed = []

    def suspect(value):
        if isinstance(value, dict):
            return any(suspect(item) for item in value.values())
        if isinstance(value, list):
            return any(suspect(item) for item in value)
        return isinstance(value, str) and any(_looks_like_secret(fragment, value) for fragment in fragments)

    for field in ("memory_candidates", "claim_updates"):
        items = getattr(intent, field, [])
        if not isinstance(items, list):
            continue
        retained = []
        for index, item in enumerate(items):
            if suspect(item):
                removed.append(f"{field}[{index}]")
            else:
                retained.append(item)
        setattr(intent, field, retained)
    updates = getattr(intent, "relationship_updates", {})
    if isinstance(updates, dict):
        retained = {}
        for name, item in updates.items():
            if suspect(item):
                removed.append("relationship_updates")
            else:
                retained[name] = item
        intent.relationship_updates = retained
    return removed


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
    # Explicit reader-only inserts cannot ride inside a public observation.
    # Include declarative offscreen viewpoint clauses, not just a labelled
    # parenthesis. This is still not a general unseen-world fact classifier.
    camera_text = re.sub(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"', "", text)
    labelled_camera = re.search(
        r"(?:[（(]\s*|^|\n)\s*(?:镜头之?外|幕后镜头|仅(?:供)?读者可见)\s*[:：]", camera_text)
    offscreen_pattern = (
        r"(?:从|在)[^。！？!?；;\n]{0,24}(?:无法看见|看不见|无法看到|看不到)的"
        r"[^。！？!?；;\n]{1,12}(?:上|里|内|中)[，,:：]"
        r"|(?:在|从)(?:众人|所有人|在场(?:者|人物|角色))的?视线(?:之)?外[，,:：]"
        r"|(?:在|从)(?:无人|没人|在场者都未|众人都未)(?:察觉|看见|注意到)的"
        r"[^。！？!?；;\n]{1,12}(?:上|里|内|中)[，,:：]"
    )
    # Unseen does not mean imperceptible: a sound/smell can reach the actors.
    # Check each viewpoint separately; one audible clause must not exempt a
    # later private visual insert. Explicit reader-only labels stay forbidden.
    offscreen_viewpoint = any(
        not re.match(r"\s*(?:传来|响起|飘来|散发|涌来)", camera_text[match.end():])
        for match in re.finditer(offscreen_pattern, camera_text)
    )
    if visibility == "public" and (labelled_camera or offscreen_viewpoint):
        issues.append(DialogueQualityIssue(
            "public_private_camera",
            "public 旁白含读者专属镜头；删除在场角色无法观察的插入，保留可见后果。不能把包内屏幕或幕后信息当作全员知识。",
            hard=True,
        ))
    recent = [
        message.speech for message in state.history[-12:]
        if message.kind == "narration" and message.speech.strip()
    ]
    if any(
        _similar(text, previous, 0.82)
        or _reordered_narration(text, previous)
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
        r"(?:点名(?:者)?|当前发言者|下一位(?:发言者|玩家|参与者)|轮到)\s*[:：]?\s*[“\"‘']?"
        r"([A-Za-z][A-Za-z0-9_.-]{1,39}|[\u4e00-\u9fff]{2,4})",
        text,
    )
    selected_names += re.findall(r"[“\"‘']([A-Za-z][A-Za-z0-9_.-]{1,39}|[\u4e00-\u9fff]{2,4})[”\"’']的座位", text)
    selected_names += re.findall(r"([\u4e00-\u9fff]{2,3})(?:喉结|深吸一口气|盯着屏幕)", text)
    for roster in re.findall(r"(?:存活名单|存活人员名单|未发言者)[：:（(]([^。；\n）)]+)", text):
        selected_names.extend(n.strip(" 、， “ ” \" ") for n in re.split(r"[、，,]", roster))
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

    if visibility == "audience_only" and not state.ended and not (
        getattr(state.world_spec, "audience_policy", "limited") == "omniscient"
        and getattr(state.world_spec, "reveal_policy", "preserve_suspense") == "allow_reveal"
    ):
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
