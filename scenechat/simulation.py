import json
import os
import re
from typing import Optional, Protocol

from .context import build_agent_view, director_context
from .dialogue_quality import inspect_dialogue_intent, quality_retry_instruction
from .dialogue_policy import safe_obligation_fallback
from .models import AgentState, Message, SimulationState
from .interventions import (
    active_guidance,
    guidance_context,
    intervention_message,
    mark_guidance_applied,
    pending_direct_event,
)
from .pacing import (
    pacing_context,
    required_beats_resolved,
    should_insert_narration,
    validate_resolved_beats,
)
from .providers import get_simulation_llm
from .runtime import Intent, IntentResolver
from .scheduler import SimulationScheduler


MAX_VISIBLE_OBSERVATIONS = 15


def _single_retry_setting(name: str, default: str = "1") -> int:
    try:
        return max(0, min(int(os.getenv(name, default)), 1))
    except (TypeError, ValueError):
        return int(default)


class AgentKnowledge(Protocol):
    def retrieve_for_agent(
        self,
        agent_name: str,
        query: str,
        top_k: int = 4,
        *,
        role: str = "",
        location: str = "",
    ) -> str:
        ...

    def retrieve_for_narrator(
        self,
        query: str,
        top_k: int = 6,
        include_private: bool = False,
    ) -> str:
        ...


def build_agent_prompt(
    state: SimulationState,
    agent: AgentState,
    retrieved_context: str,
) -> str:
    view = build_agent_view(state, agent, retrieved_context)
    relationship_dimensions = json.dumps(
        state.relationship_dimensions,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return f"""你正在扮演社会模拟实验中的角色“{agent.name}”。

你不是全知叙述者。你只能依据下面明确提供的信息判断，绝不能假定自己知道其他角色的私密动机、秘密经历或未被观察到的事件。

{view.render()}

请严格站在“{agent.name}”的有限视角中推进一轮行动。行动和发言必须符合其身份、目标、已知信息与社会处境，不要解释创作过程，不要替其他角色行动。

【自然对话规则】
1. 如果“需要优先处理的回应”非空，本轮应先回应其中最紧迫的一项；填写对应 thread_id、reply_to_obligation_id 和 reply_to_event_id。可以回答、质疑、回避或拒绝，但不能像没听见一样另起话题。回避表示已经回应但没有解决，不要伪装成完整回答。
   标为“可选择插话”的内容不是强制回应；只有当它与当前目标、关系或秘密风险确实相关时才插话，避免所有人逢点名必抢话。
2. 每轮只推进一个主要意图。优先对最近的具体言行作出反应，不要重新介绍人物、世界背景或复述双方已经知道的事实。
3. 遵循语言画像，但不要机械重复口癖。让句长、礼貌、直接程度、情绪外显和信息披露方式体现人物差异。
4. 角色通常不会把完整动机、秘密和推理过程直接说出口。允许潜台词、停顿、试探、反问、转移、沉默，以及动作与台词不完全一致。
5. 除非题材或当前情境要求正式陈述，speech 通常控制在一至三句；没有必要说话时可以只行动或保持沉默。
6. 本场景的关系维度为：{relationship_dimensions}。relationship_updates.facets 只能使用这些维度 ID；它们是题材相关的主观关系，不是固定的狼人杀式猜疑参数，也不能代替生命值、距离或胜负等客观战斗状态。每次变化必须引用新观察到的历史事件。
7. 信息披露压力按议题隔离。某个议题压力较高时，角色更难完全无视该议题的追问，但仍可以有限回答、转移、撒谎或反问；不要因此公开其他议题的秘密。
8. 他人的发言首先只是主张，不是客观事实。只有权威事件才能直接成为 verified；普通听闻、观察和推断用 claim_updates 保存来源。used_memory_ids 只填写本轮确实使用的已知事实、记忆或主张 ID。

你只提交 Intent，不直接修改世界状态。action_type、ability 和 target 必须来自上面的当前阶段、能力与在场角色；不能凭空宣布自己获胜、获得能力、知道秘密或强迫他人完成重大决定。private_reason 只解释本轮决策，不会公开，也不会自动写入长期记忆。

只输出一个 JSON 对象，不要使用 Markdown 代码块：
{{
  "action": "动作描述",
  "speech": "角色说出的话，没有台词时可为空字符串",
  "action_type": "speak|act|observe|pass|move|vote|inspect|protect|eliminate|heal|poison|场景定义的类型",
  "target": "目标角色或地点；不需要目标时为空",
  "ability": "能力 ID 或名称；不使用能力时为空",
  "private_reason": "该角色不会说出口的一句理由",
  "relationship_updates": {{
    "其他角色姓名": {{
      "facets": {{"本场景声明的关系维度 ID": 0.0}},
      "reason_event_id": "造成变化的可见历史 event_id",
      "private_note": "该事件为何改变了判断",
      "summary": "可选：更新后的简短主观关系描述"
    }}
  }},
  "addressed_to": ["本轮明确对话或行动指向的角色姓名"],
  "mentioned_agents": ["被谈及、可能因此选择插话，但本轮并非直接对话对象的角色"],
  "reply_to_event_id": "正在回应的可见消息 event_id；没有时为空",
  "thread_id": "正在延续的可见议题 thread_id；新议题或无关行动时为空",
  "reply_to_obligation_id": "正在回应的 obligation_id；没有时为空",
  "obligation_resolution": "responded|satisfied|withdrawn；回避和反问只能填 responded",
  "conversation_move": "statement|answer|question|request|challenge|deflect|support|reveal|acknowledge|silence",
  "urgency": 0.0,
  "short_term_state": {{
    "emotion": {{"label": "本轮后的具体情绪", "intensity": 0.0}},
    "conversation_goal": "接下来几轮希望通过交流达成什么",
    "disclosure_pressure_delta": 0.0,
    "commitments_add": ["本轮新作出的、之后需要履行的承诺"],
    "commitments_resolve": ["本轮已经履行的既有承诺，文字必须与列表一致"]
  }},
  "memory_candidates": [{{
    "type": "claim|clue|commitment|relationship_evidence|revelation|decision",
    "content": "本轮值得跨多轮保留的一条具体信息；普通动作和泛泛心理不要填写",
    "importance": 1,
    "related_agents": ["与这条记忆直接相关的角色"],
    "source_event_id": "信息来自某条可见历史消息时填写，否则为空"
  }}],
  "used_memory_ids": ["本轮实际使用的可见事实、记忆或认知记录 ID"],
  "claim_updates": [{{
    "content": "值得保留的具体主张或认知，不要把普通台词全部复制进来",
    "source_event_id": "来源事件；自身本轮新主张可为空",
    "epistemic_status": "heard|observed|inferred|believed|verified|disputed|disproved",
    "confidence": 0.5,
    "related_agents": ["相关角色"],
    "supersedes": "被本条认知修正或推翻的既有 belief ID；没有时为空"
  }}],
  "expected_effect": "角色期望发生什么，不代表一定成功",
  "proposed_patch": []
}}
"""


def parse_agent_response(raw: str) -> Optional[tuple[str, str]]:
    raw = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        payload = json.loads(raw)
        action = str(payload.get("action") or "说道").strip()
        speech = str(payload.get("speech") or "").strip()
        return (action, speech) if speech else None
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Backward-compatible parsing for the original one-line format.
    match = re.match(r"^(?:[^:：]+[:：])?\s*(.*?)\s+[“\"]?(.*?)[”\"]?$", raw)
    if not match:
        return None
    action, speech = (part.strip() for part in match.groups())
    return (action or "说道", speech) if speech else None


def parse_agent_intent(raw: str) -> Optional[dict]:
    """Parse the stateful response while retaining the legacy text fallback."""

    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = parse_agent_response(raw)
        if parsed is None:
            return None
        return {
            "action": parsed[0],
            "speech": parsed[1],
            "action_type": "speak",
            "target": "",
            "ability": "",
            "private_reason": "",
            "expected_effect": "",
            "proposed_patch": [],
            "relationship_updates": {},
            "addressed_to": [],
            "mentioned_agents": [],
            "reply_to_event_id": "",
            "thread_id": "",
            "reply_to_obligation_id": "",
            "obligation_resolution": "",
            "conversation_move": "statement",
            "urgency": 0.0,
            "short_term_state": {},
            "memory_candidates": [],
            "used_memory_ids": [],
            "claim_updates": [],
        }
    if not isinstance(payload, dict):
        return None
    action = str(payload.get("action") or "观察局势").strip()
    speech = str(payload.get("speech") or "").strip()
    if not action and not speech:
        return None
    return {
        "action": action,
        "speech": speech,
        "action_type": str(payload.get("action_type") or "speak").strip(),
        "target": str(payload.get("target") or "").strip(),
        "ability": str(payload.get("ability") or "").strip(),
        "private_reason": str(payload.get("private_reason") or payload.get("memory") or "").strip(),
        "expected_effect": str(payload.get("expected_effect") or "").strip(),
        "proposed_patch": payload.get("proposed_patch")
        if isinstance(payload.get("proposed_patch"), list)
        else [],
        "relationship_updates": payload.get("relationship_updates")
        if isinstance(payload.get("relationship_updates"), dict)
        else {},
        "addressed_to": payload.get("addressed_to")
        if isinstance(payload.get("addressed_to"), list)
        else [],
        "mentioned_agents": payload.get("mentioned_agents")
        if isinstance(payload.get("mentioned_agents"), list)
        else [],
        "reply_to_event_id": str(payload.get("reply_to_event_id") or "").strip(),
        "thread_id": str(payload.get("thread_id") or "").strip(),
        "reply_to_obligation_id": str(payload.get("reply_to_obligation_id") or "").strip(),
        "obligation_resolution": str(payload.get("obligation_resolution") or "").strip(),
        "conversation_move": str(payload.get("conversation_move") or "statement").strip(),
        "urgency": payload.get("urgency", 0.0),
        "short_term_state": payload.get("short_term_state")
        if isinstance(payload.get("short_term_state"), dict)
        else {},
        "memory_candidates": payload.get("memory_candidates")
        if isinstance(payload.get("memory_candidates"), list)
        else [],
        "used_memory_ids": payload.get("used_memory_ids")
        if isinstance(payload.get("used_memory_ids"), list)
        else [],
        "claim_updates": payload.get("claim_updates")
        if isinstance(payload.get("claim_updates"), list)
        else [],
    }


def build_narrator_prompt(
    state: SimulationState,
    retrieved_context: str,
    visibility: str,
) -> str:
    if visibility == "public":
        mode_instruction = (
            "本轮必须生成 public 事件：只能描述在场角色能够直接观察到的环境变化或中立事件，"
            "不得出现任何角色的隐藏身份、私人想法或未公开计划。"
        )
        recent_history = state.get_recent_history(12, public_only=True)
    else:
        mode_instruction = (
            "本轮必须生成 audience_only 旁白：可以给读者补充镜头信息或未被角色察觉的事实，"
            "但这些内容不会成为角色知识。"
        )
        recent_history = state.get_recent_history(12)

    return f"""你是互动故事的场景导演与旁白，不扮演任何一个角色。

【实验设定与可用背景】
{retrieved_context}

【当前场景】
{state.scene}

【当前阶段、公共状态与结束条件】
{state.public_state_summary()}

【近期叙事与公开行动】
{recent_history}

【导演节奏与剧情弧】
{pacing_context(state)}

【已确认的用户导演指令】
{guidance_context(state)}

请生成一个符合原题材的简短叙事事件，用于补充环境、节奏、动作结果、中立事件或面向读者的镜头信息。不要替角色说台词，不要强迫角色作出重大决定，不要突然转换题材，也不要无依据加入 AI、未来科技或宏大阴谋。

{mode_instruction}

只输出一个 JSON 对象，不要使用 Markdown：
{{
  "narration": "一至三句自然的叙述",
  "visibility": "{visibility}",
  "location": "事件只发生在某个地点时填写；全局广播或读者镜头可为空",
  "state_updates": {{"公共状态变量": "环境事件或规则裁决造成的新值"}},
  "resolved_beat_ids": ["只有本轮已经实际完成的当前可推进节点 ID；不要把刚埋下的线索算作完成"],
  "tension": 0.0,
  "end_signal": false,
  "end_reason": "只有确实满足结束条件时填写"
}}
"""


def parse_narrator_response(raw: str) -> Optional[tuple[str, str]]:
    raw = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        payload = json.loads(raw)
        narration = str(payload.get("narration") or "").strip()
        visibility = str(payload.get("visibility") or "public").strip()
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None
    if not narration:
        return None
    if visibility not in {"public", "audience_only"}:
        visibility = "public"
    return narration, visibility


def parse_narrator_event(raw: str) -> Optional[dict]:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    parsed = parse_narrator_response(text)
    if parsed is None:
        return None
    updates = payload.get("state_updates")
    return {
        "narration": parsed[0],
        "visibility": parsed[1],
        "state_updates": updates if isinstance(updates, dict) else {},
        "end_signal": bool(payload.get("end_signal", False)),
        "end_reason": str(payload.get("end_reason") or "").strip(),
        "location": str(payload.get("location") or "").strip(),
        "resolved_beat_ids": [
            str(item) for item in payload.get("resolved_beat_ids") or []
        ] if isinstance(payload.get("resolved_beat_ids"), list) else [],
        "tension": payload.get("tension"),
    }


def simulate_next_turn(
    state: SimulationState,
    knowledge_base: AgentKnowledge,
    llm=None,
    *,
    agent: AgentState | None = None,
    resolver: IntentResolver | None = None,
) -> Optional[Message]:
    agent = agent or state.next_agent()
    query = (
        f"当前场景：{state.scene}\n"
        f"当前角色：{agent.name}\n"
        f"角色目标：{'；'.join(agent.goals)}\n"
        f"可选核心信念：{'；'.join(agent.core_beliefs) or '无额外设定'}\n"
        f"近期观察：{agent.recent_observations(5)}\n"
        f"当前对话目标：{agent.current_conversation_goal or '依据长期目标判断'}\n"
        f"待回应对象：{agent.last_addressed_by or '无'}\n"
        f"当前各议题披露压力：{agent.disclosure_pressure_by_thread or {'general': agent.disclosure_pressure}}\n"
        f"活跃议题：{'；'.join(thread.topic for thread in state.active_threads_for(agent.name)[:4]) or '无'}\n"
        f"可选择插话数：{len(agent.conversation_opportunities)}"
    )
    try:
        retrieved_context = knowledge_base.retrieve_for_agent(
            agent.name,
            query,
            role=agent.role,
            location=agent.current_location,
        )
    except TypeError:
        # Compatibility for older custom knowledge-base implementations.
        retrieved_context = knowledge_base.retrieve_for_agent(agent.name, query)
    prompt = build_agent_prompt(state, agent, retrieved_context)
    active_llm = llm or get_simulation_llm()
    active_resolver = resolver or IntentResolver()
    retries = max(
        _single_retry_setting("SIMULATION_PARSE_RETRIES"),
        _single_retry_setting("SIMULATION_QUALITY_RETRIES"),
    )
    rejection = ""
    for attempt in range(retries + 1):
        attempt_prompt = prompt
        if rejection:
            attempt_prompt += (
                "\n\n【上一次 Intent 被拒绝】\n"
                f"{rejection}\n请保持角色目标不变，改为提交一项当前阶段合法的 Intent。"
            )
        response = active_llm.complete(attempt_prompt, max_tokens=520)
        parsed = parse_agent_intent(response.text)
        if parsed is None:
            rejection = "输出不是合法的 Intent JSON"
            continue
        intent = Intent.from_mapping(agent.name, parsed)
        resolution = active_resolver.resolve(state, intent)
        if not resolution.accepted:
            rejection = resolution.reason
            continue
        quality_issues = inspect_dialogue_intent(state, agent, intent)
        if quality_issues:
            rejection = (
                "Intent 通过规则校验，但未通过对话质量门：\n"
                f"{quality_retry_instruction(quality_issues)}\n"
                "保持人物目标和行动方向不变，只修正上述问题。"
            )
            should_retry = attempt < retries
            state.record_dialogue_quality_issues(
                [issue.code for issue in quality_issues],
                retried=should_retry,
            )
            if should_retry or any(issue.hard for issue in quality_issues):
                continue
        state.record_generation_success()
        return _message_from_resolution(state, agent, intent, resolution)

    phase = state.phase_specs.get(state.current_phase)
    allowed_actions = list(getattr(phase, "allowed_action_types", []) or [])
    pending = agent.pending_intents[-1] if agent.pending_intents else None
    if pending and (not allowed_actions or "speak" in allowed_actions):
        fallback_intent = safe_obligation_fallback(agent, pending, rejection)
        fallback_resolution = active_resolver.resolve(state, fallback_intent)
        if fallback_resolution.accepted and fallback_intent.reply_to_event_id:
            state.record_generation_success()
            return _message_from_resolution(
                state,
                agent,
                fallback_intent,
                fallback_resolution,
            )
    fallback_type = next(
        (candidate for candidate in ("pass", "observe", "speak", "act") if candidate in allowed_actions),
        "" if allowed_actions else "pass",
    )
    if fallback_type:
        fallback_intent = Intent(
            actor=agent.name,
            action_type=fallback_type,
            action="暂不采取额外行动，继续观察局势。",
            private_reason=f"先前的行动意图未通过规则校验：{rejection}",
        )
        fallback_resolution = active_resolver.resolve(state, fallback_intent)
        if fallback_resolution.accepted:
            state.record_generation_success()
            return _message_from_resolution(
                state,
                agent,
                fallback_intent,
                fallback_resolution,
            )
    state.record_generation_failure()
    return None


def _message_from_resolution(state, agent, intent, resolution) -> Message:
    return Message(
        speaker=agent.name,
        action=intent.action,
        speech=intent.speech,
        turn=state.turn_count + 1,
        visibility=resolution.visibility,
        visibility_scopes=resolution.visibility_scopes,
        location=resolution.location,
        participants=resolution.participants,
        individual_observations=resolution.individual_observations,
        state_updates=resolution.patch.public_updates(),
        memory="",
        relationship_updates=intent.relationship_updates,
        authoritative=True,
        state_patch=resolution.patch.operations,
        intent=intent.to_dict(),
    )


def simulate_narration(
    state: SimulationState,
    knowledge_base: AgentKnowledge,
    llm=None,
    *,
    resolver: IntentResolver | None = None,
    forced_visibility: str | None = None,
) -> Optional[Message]:
    # Public narration never receives private character documents. Reader-only
    # narration may use them, but is never broadcast into character observations.
    visibility = forced_visibility or (
        "public" if state.narration_count % 2 == 0 else "audience_only"
    )
    public_only = visibility == "public"
    query = (
        f"当前场景：{state.scene}\n"
        f"近期进展：{state.get_recent_history(8, public_only=public_only)}\n"
        "检索适合推动当前题材的环境规则、角色秘密或事件线索。"
    )
    retrieved_context = knowledge_base.retrieve_for_narrator(
        query,
        include_private=not public_only,
    )
    narrator_context = retrieved_context
    if visibility == "audience_only":
        narrator_context = f"{director_context(state)}\n\n【检索背景】\n{retrieved_context}"
    prompt = build_narrator_prompt(state, narrator_context, visibility)
    active_llm = llm or get_simulation_llm()
    parsed = None
    retries = _single_retry_setting("SIMULATION_PARSE_RETRIES")
    for attempt in range(retries + 1):
        attempt_prompt = prompt if attempt == 0 else (
            prompt + "\n\n上一次输出无法解析。只重新输出一个符合 schema 的完整 JSON 对象。"
        )
        response = active_llm.complete(attempt_prompt, max_tokens=360)
        parsed = parse_narrator_event(response.text)
        if parsed is not None:
            break
    if parsed is None:
        state.record_generation_failure()
        return None
    resolved_beat_ids = validate_resolved_beats(state, parsed["resolved_beat_ids"])
    resolution = (resolver or IntentResolver()).resolve_director_event(
        state,
        narration=parsed["narration"],
        visibility=visibility,
        proposed_updates=parsed["state_updates"],
        end_signal=parsed["end_signal"] and required_beats_resolved(state, resolved_beat_ids),
        end_reason=parsed["end_reason"],
        location=parsed["location"],
    )
    state.record_generation_success()
    applied_guidance_ids = mark_guidance_applied(state)
    intent_payload = resolution.intent.to_dict()
    intent_payload["arc_updates"] = {
        "resolved_beat_ids": resolved_beat_ids,
        "tension": parsed["tension"],
    }
    if applied_guidance_ids:
        intent_payload["intervention_ids"] = applied_guidance_ids
    return Message(
        speaker="旁白",
        action="场景推进",
        speech=parsed["narration"],
        turn=state.turn_count + 1,
        kind="narration",
        visibility=visibility,
        visibility_scopes=resolution.visibility_scopes,
        location=resolution.location,
        state_updates=resolution.patch.public_updates(),
        end_signal=resolution.end_signal,
        end_reason=resolution.end_reason,
        authoritative=True,
        state_patch=resolution.patch.operations,
        intent=intent_payload,
    )


def simulate_next_event(
    state: SimulationState,
    knowledge_base: AgentKnowledge,
    llm=None,
    *,
    scheduler: SimulationScheduler | None = None,
    resolver: IntentResolver | None = None,
) -> Optional[Message]:
    director_event = pending_direct_event(state)
    if director_event is not None:
        return intervention_message(state, director_event)
    guidance_waiting = any(item.status == "pending" for item in active_guidance(state))
    should_narrate = guidance_waiting or should_insert_narration(state)
    if should_narrate:
        narration = simulate_narration(
            state,
            knowledge_base,
            llm=llm,
            resolver=resolver,
            forced_visibility="public" if guidance_waiting else None,
        )
        if narration is not None:
            return narration
    decision = (scheduler or SimulationScheduler()).decide(state)
    if decision.kind in {"narration", "event"}:
        return simulate_narration(
            state,
            knowledge_base,
            llm=llm,
            resolver=resolver,
            forced_visibility="public" if decision.kind == "event" else None,
        )
    return simulate_next_turn(
        state,
        knowledge_base,
        llm=llm,
        agent=state.agents[decision.actor_name],
        resolver=resolver,
    )
