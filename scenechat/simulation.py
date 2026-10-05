import json
import inspect
import re
from typing import Optional, Protocol

from .context import build_agent_view, director_context
from .config import config_int
from .errors import SceneChatError
from .dialogue_quality import (
    inspect_dialogue_intent,
    inspect_narration_event,
    quality_retry_instruction,
)
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
from .scenario import extract_json_object
from .scheduler import SimulationScheduler
from .telemetry import complete
from .mechanics import visible_entities, action_context
from .recovery import bounded_operation
from .prompt_library import prompt_contract, actor_worklist, narrator_facts


MAX_VISIBLE_OBSERVATIONS = 15


def _single_retry_setting(key: str, default: int = 1) -> int:
    return config_int("simulation", key, default, minimum=0, maximum=1)


def _token_budget(key: str, default: int, minimum: int, maximum: int) -> int:
    return config_int("simulation", key, default, minimum=minimum, maximum=maximum)


def _repair_vote_rule_reference(state: SimulationState, agent: AgentState, intent: Intent) -> bool:
    """Repair only a mismatched rule ID when the ballot has one legal route."""
    if (getattr(state.world_spec, "execution_version", 1) != 2
            or intent.action_type != "vote" or not intent.rule_id
            or not intent.target or intent.ability):
        return False
    if IntentResolver._matching_rule(state, agent, intent) is not None:
        return False
    from .mechanics import records_vote
    from .role_selectors import matches_role
    candidates = [
        rule for rule in state.rules
        if rule.action_type == "vote" and records_vote(rule)
        and (not rule.phases or state.current_phase in rule.phases)
        and matches_role(agent.role, rule.allowed_roles)
        and not IntentResolver._validate_target(state, agent, intent, None, rule)
    ]
    if len(candidates) != 1:
        return False
    previous = intent.rule_id
    intent.rule_id = candidates[0].id
    state.record_structured_output_issue("vote_rule_reference_repaired", retried=False)
    if state.model_requests:
        state.model_requests[-1]["local_rule_repair"] = {"from": previous, "to": intent.rule_id}
    return True


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
    phase = state.phase_specs.get(state.current_phase)
    phase_actions = "、".join(phase.allowed_action_types) if phase and phase.allowed_action_types else "由可用规则决定"
    exit_rule_id = str(state.last_scheduler_decision.get("phase_exit_rule_id") or "")
    phase_exit_instruction = (
        f"【本轮阶段推进】讨论已达到当前节奏允许的篇幅，继续换措辞争同一件事不会产生新信息。"
        f"本轮请选择当前可执行的规则 {exit_rule_id}，用角色自己的话或动作发起阶段切换；"
        "不要声称投票或结算已经完成。若还有待回应问题，可简短交代自己的立场，但不能用新反问拖延阶段。"
        if exit_rule_id else ""
    )

    return f"""{prompt_contract('actor')}
你正在扮演社会模拟实验中的角色“{agent.name}”。

你不是全知叙述者。你只能依据下面明确提供的信息判断，绝不能假定自己知道其他角色的私密动机、秘密经历或未被观察到的事件。

{view.render()}

【推进工作单】
{actor_worklist(state, agent)}

【可用的非人物目标】
地点：{json.dumps(state.locations, ensure_ascii=False)}
物品/提案（使用声明的 ID）：{json.dumps(visible_entities(state, agent), ensure_ascii=False)}
本阶段允许的 action_type：{phase_actions}
当前允许的场景行动（不含尚未获取的结果）：
{action_context(state, agent)}

{phase_exit_instruction}

请严格站在“{agent.name}”的有限视角中推进一轮行动。行动和发言必须符合其身份、目标、已知信息与社会处境，不要解释创作过程，不要替其他角色行动。

【人物决策镜头——先是具体的人，再是身份与表达画像】
已设定的性格倾向：{agent.personality or '未单列；从完整人物档案中提取，不能凭职业套用。'}
已设定的惯常决策方式：{agent.decision_logic or '未单列；从完整人物档案与当前目标判断。'}
“当前事项”只说明外部问题，不规定此人必须合作。先判断眼前哪句话、行动或后果真正触动了自己；此刻能确定什么、怕失去什么、在谁面前不愿丢脸、愿意说到哪一步。再由性格、关系、经历影响选择。性别、文化、社会处境和职业按用户设定真实存在，但不是每句台词要表演的标签；职业主要影响可用知识与注意力，不强迫使用术语。没有明确风险时不虚构惊恐；有高风险时也不让所有人同样冷静或同样慌张。
同一事项下，不同人物可以给出相反选择，也可以承认不知道、暂时强撑、说一半收住或事后补救。不要默认人人善辩、毒舌、理性、礼貌或寻求共识。private_reason 只写这个人此刻的具体顾虑或冲动及其选择，不必写漂亮的策略总结；它是不会说出口的材料，action/speech 不要复述它。
如果当前是陌生人初遇，别凭空假设彼此有旧交、共同记忆或已有信任。即使主线涉及生死，也不意味着每人开口都在诊断敌人：规则真伪、眼前安全、身体反应、逃出去的念头、亲友牵挂或对某个人的第一印象，都可能改变其行动与说话。只挑眼下有关的东西，不能机械给每人塞一条无关家常；少数确有理由的人可立即尖锐，多数人可先摸底、试探或保持距离。

【自然对话规则】
1. 如果“需要优先处理的回应”非空，本轮应先回应其中最紧迫的一项；填写对应 thread_id、reply_to_obligation_id 和 reply_to_event_id。可以回答、质疑、回避或拒绝，但不能像没听见一样另起话题。回避表示已经回应但没有解决，不要伪装成完整回答。
   若对方问的是“你现在怎么选择/愿意做什么”，先给出此人此刻的选择、明确拒绝或行动条件，再提出必要的反问；只有反问并未完成原请求。对已有请求的反问默认只是该事项的一部分，不会再强制生成一轮新的提问义务。没有回答或不知道时 obligation_resolution 填 responded，不得填 satisfied；不能替发起人撤回请求。
   标为“可选择插话”的内容不是强制回应；只有当它与当前目标、关系或秘密风险确实相关时才插话，避免所有人逢点名必抢话。
2. 每轮只推进一个主要意图。优先对最近的具体言行作出反应，不要重新介绍人物、世界背景或复述双方已经知道的事实。
3. 先让性格影响选择、冒险程度和信息披露，再让语言画像影响措辞。说话应先接住眼前具体的人和事：能直接答就答，不能确定就说清自己不确定的部分。不要把每次回应写成概括人性、秩序、命运等主题的金句，也不要反复使用职业比喻、人物标志句或同一种挑衅动作；档案中的表达习惯不是每轮必演的台词。
4. 角色通常不会把完整动机、秘密和推理过程直接说出口。情绪可以通过答得太快、删掉半句解释、转移对象、短暂失态或刻意平静影响表达，但只在此刻有原因时出现；不要机械加入结巴、语气词和身体小动作。纯发言时 action 可以简写“回应”“说话”，无需每次撑桌、盯视、拍桌或描写表情。允许潜台词、试探、反问、沉默，以及动作与台词不完全一致。高明的人也可能说一句很普通的话。
5. 除非题材或当前情境要求正式陈述，speech 通常控制在一至三句；句长和完整程度应随情境变化，不必每轮都有反讽、收尾句或“漂亮的”比喻。没有必要说话时可以只行动或保持沉默。面向全场提问、请求或质疑时，address_scope 必须填 all_present；只对具体角色说话时填 specific。
6. 本场景的关系维度为：{relationship_dimensions}。relationship_updates.facets 只能使用这些维度 ID；它们是题材相关的主观关系，不是固定的狼人杀式猜疑参数，也不能代替生命值、距离或胜负等客观战斗状态。每次变化必须引用新观察到的历史事件。
7. 信息披露压力按议题隔离。某个议题压力较高时，角色更难完全无视该议题的追问，但仍可以有限回答、转移、撒谎或反问；不要因此公开其他议题的秘密。
8. 他人的发言首先只是主张，不是客观事实。只有权威事件才能直接成为 verified；普通听闻、观察和推断用 claim_updates 保存来源。used_memory_ids 只填写本轮确实使用的已知事实、记忆或主张 ID。想引用别人“刚才说过”的话，必须能从可见历史或有来源的记忆找到；不要为了让推理显得聪明而编造旧发言。对别人瞳孔、呼吸、心率、手势等细节也是一样：没有亲眼可见的事件或设定中的测量能力，就不能报出具体测量结果；最多说出可见行为带给自己的不确定印象。
9. 已经达成的具体做法不需要每人再复述确认。如果轮到你且能执行第一步，就用当前合法行动实际做；若只能提议而不能做，说明尚缺的条件。不要把“我去检查”写成“已查清结果”，也不要用未经设定的具体共同往事、年份、事故或关键证据补出戏剧效果；没有把握时用不确定的日常说法。

你只提交 Intent，不直接修改世界状态。普通发言的 action_type 必须原样填写英文枚举 `speak`，普通行动/观察/跳过分别填写 `act`、`observe`、`pass`，不得翻译或自造“公开行动”“对话行动”等值。专用 action_type、ability 和 target 必须逐字来自上面的当前阶段、能力与在场角色；没有使用能力时省略 ability，不能把能力说明或期望效果填成能力名。不能凭空宣布自己获胜、获得能力、知道秘密或强迫他人完成重大决定。private_reason 只解释本轮决策，不会公开，也不会自动写入长期记忆。

只输出一个紧凑 JSON 对象，不要使用 Markdown 代码块。private_reason 请先用一句短话写出此刻实际在意的东西，再生成公开行为；不要把这句私心照搬到台词。action、speech、action_type 必填；其他字段没有实际变化时可以省略，不要为了字段齐全输出大段空数组：
{{
  "private_reason": "此刻具体的顾虑、欲望或冲动；不是漂亮的策略总结",
  "action": "动作描述",
  "speech": "角色台词，没有时为空字符串",
  "action_type": "当前阶段允许的行动类型",
  "rule_id": "需要特定规则效果时填可用规则 ID；普通发言可省略",
  "target": "目标角色或地点",
  "ability": "能力 ID 或名称",
  "address_scope": "none|specific|all_present",
  "addressed_to": ["直接对话对象"],
  "mentioned_agents": ["被谈及但不是对话对象的角色"],
  "reply_to_event_id": "正在回应的 event_id",
  "thread_id": "正在延续的 thread_id",
  "reply_to_obligation_id": "正在回应的 obligation_id",
  "obligation_resolution": "responded|satisfied|withdrawn",
  "conversation_move": "statement|answer|question|request|challenge|deflect|support|reveal|acknowledge|silence",
  "urgency": 0.0,
  "short_term_state": {{"emotion":{{"label":"情绪","intensity":0.0}},"conversation_goal":"交流目标","disclosure_pressure_delta":0.0,"commitments_add":[],"commitments_resolve":[]}},
  "relationship_updates": {{"角色名":{{"facets":{{"维度 ID":0.0}},"reason_event_id":"可见 event_id","private_note":"变化依据","summary":"关系摘要"}}}},
  "memory_candidates": [{{"type":"claim|clue|commitment|relationship_evidence|revelation|decision","content":"跨轮信息","importance":1,"related_agents":[],"source_event_id":""}}],
  "used_memory_ids": ["实际使用的事实、记忆或认知 ID"],
  "claim_updates": [{{"content":"具体主张","source_event_id":"","epistemic_status":"heard|observed|inferred|believed|verified|disputed|disproved","confidence":0.5,"related_agents":[],"supersedes":""}}],
  "expected_effect": "期望效果，不代表实际结果"
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
        payload = extract_json_object(text)
    except (ValueError, json.JSONDecodeError, TypeError):
        # A response that started as JSON but was truncated must not be
        # reinterpreted as the legacy prose format.
        if text.lstrip().startswith(("{", "[")):
            return None
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
            "address_scope": "none",
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
        "rule_id": str(payload.get("rule_id") or "").strip(),
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
        "address_scope": str(payload.get("address_scope") or "none").strip(),
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

    reveal_instruction = (
        "用户允许全知读者镜头揭示秘密；只在 audience_only 镜头中使用，不得写入角色知识。"
        if getattr(state.world_spec, "audience_policy", "limited") == "omniscient" and getattr(state.world_spec, "reveal_policy", "preserve_suspense") == "allow_reveal"
        else "在身份公开揭晓或结算前，只能给出有多种解释的线索，不得直接确认隐藏身份。"
    )
    return f"""{prompt_contract('narrator')}

【引擎事实】
{narrator_facts(state)}

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
优先回应最近已执行的行动，呈现有限、可观察的直接后果；不要为制造张力恰好安排新路人带来同类问题，或连续升级同一处天气、故障、危险。没有值得写的新变化且本阶段不强制旁白时，允许跳过。未设定的具体共同往事不能当成确定事实。
若角色正在查找责任、病因或秘密来源，不要临时发明撕痕、监控、物证或测量结果指向某人；保留未知，只结算已经有依据的可见结果。
涉及人数、票数和阶段时，以【引擎事实】为准；public_active_count 为 null 表示不能公开推断人数。投票未完成时不要宣布本轮结果、清空票数或虚构界面提示。未设定终端、灯光、按钮、座椅机构等具体装置时，不要把它们当成确定存在的系统反馈；可直接叙述已提交的结果。
也不要替角色补做未提交的小动作，如包好杯子、打开抽屉或迈出一步；只呈现已提交行动的后果或独立环境变化。
旁白不能用反复改写灯光、计时器、涟漪、沉默来占用回合；须回应刚发生的实际行动或提供可观察的新事件。规则宣读、结算等环境阶段应完成该阶段的职责，可以由中立广播说明规则或已发生的结果；不能只写气氛就声称已宣读规则或完成讨论。resolved_beat_ids 缺少实际证据时留空。

{mode_instruction}

{reveal_instruction} 不得反复用代码流、后台进程、异常同步等单一答案式暗示。不要指定下一位行动者。先检查近期历史：如果某个当前节点已经由最近的角色行动实际完成，在 resolved_beat_ids 中提议核验；不要把本句旁白自己宣布的结果当证据。

tension 必须使用 0.0—1.0 的小数比例，不能填写百分制的 25 或 100。

没有可呈现的新内容且不是必需环境阶段/待应用干预时，可只输出 {{"skip":true}}。否则只输出一个 JSON 对象，不要使用 Markdown：
{{
  "narration": "一至三句自然的叙述",
  "visibility": "{visibility}",
  "location": "事件只发生在某个地点时填写；全局广播或读者镜头可为空",
  "state_updates": {{"允许 director 修改的公共环境字段": "新值"}},
  "evidence_event_ids": ["引用已提交结果时填写来源事件 ID"],
  "resolved_beat_ids": ["本轮或最近尚未结算的已完成节点 ID；只埋下线索时不要填写"],
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
        payload = extract_json_object(raw)
        narration = str(payload.get("narration") or "").strip()
        visibility = str(payload.get("visibility") or "public").strip()
    except (ValueError, json.JSONDecodeError, AttributeError, TypeError):
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
        payload = extract_json_object(text)
    except (ValueError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("skip") is True:
        return {"skip": True}
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
        "evidence_event_ids": payload.get("evidence_event_ids") if isinstance(payload.get("evidence_event_ids"), list) else [],
    }


@bounded_operation("simulation")
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
    retrieve = knowledge_base.retrieve_for_agent
    try:
        signature = inspect.signature(retrieve)
    except (ValueError, TypeError):
        signature = None
    legacy_retrieval = False
    if signature is not None:
        try:
            signature.bind(agent.name, query, role=agent.role, location=agent.current_location)
        except TypeError:
            legacy_retrieval = True
    if legacy_retrieval:
        retrieved_context = retrieve(agent.name, query)
    else:
        retrieved_context = knowledge_base.retrieve_for_agent(
            agent.name,
            query,
            role=agent.role,
            location=agent.current_location,
        )
    prompt = build_agent_prompt(state, agent, retrieved_context)
    active_llm = llm or get_simulation_llm()
    active_resolver = resolver or IntentResolver()
    phase_exit_rule_id = str(state.last_scheduler_decision.get("phase_exit_rule_id") or "")
    retries = max(
        _single_retry_setting("parse_retries"),
        _single_retry_setting("quality_retries"),
    )
    rejection = ""
    for attempt in range(retries + 1):
        attempt_prompt = prompt
        if rejection:
            attempt_prompt += (
                "\n\n【上一次 Intent 被拒绝】\n"
                f"{rejection}\n请保持角色目标不变，改为提交一项当前阶段合法的 Intent。"
            )
        response = complete(state, active_llm,
            attempt_prompt,
            purpose="actor_intent",
            max_tokens=_token_budget("intent_max_tokens", 900, 520, 1800),
        )
        parsed = parse_agent_intent(response.text)
        if parsed is None:
            finish_reason = str(getattr(response, "finish_reason", "") or "")
            truncated = finish_reason == "length" or (
                str(response.text or "").lstrip().startswith("{")
                and not str(response.text or "").rstrip().endswith("}")
            )
            code = "intent_json_truncated" if truncated else "intent_json_invalid"
            should_retry = attempt < _single_retry_setting("parse_retries")
            state.record_structured_output_issue(code, retried=should_retry)
            rejection = (
                "Intent JSON 输出被截断。省略所有没有变化的可选字段，优先闭合一个简短 JSON 对象。"
                if truncated else
                "输出不是合法的 Intent JSON。只输出一个 JSON 对象，并省略未使用的可选字段。"
            )
            if not should_retry:
                break
            continue
        intent = Intent.from_mapping(agent.name, parsed)
        _repair_vote_rule_reference(state, agent, intent)
        resolution = active_resolver.resolve(state, intent)
        if not resolution.accepted:
            if state.model_requests:
                state.model_requests[-1]["rule_rejection_reason"] = resolution.reason[:200]
            state.record_structured_output_issue(
                "intent_rule_rejected", retried=attempt < _single_retry_setting("quality_retries")
            )
            rejection = (
                f"{resolution.reason}\n"
                f"当前阶段={state.current_phase}；允许的 action_type="
                f"{','.join(state.phase_specs[state.current_phase].allowed_action_types) if state.current_phase in state.phase_specs else '见运行状态'}。\n"
                f"可执行规则与合法目标：\n{action_context(state, agent)}"
            )
            if attempt >= _single_retry_setting("quality_retries"):
                break
            continue
        if phase_exit_rule_id and not any(
            operation.get("op") == "set_phase"
            and operation.get("value") == state.phase_specs[state.current_phase].next_phase
            for operation in resolution.patch.operations
        ):
            rejection = (
                f"本轮必须使用 rule_id={phase_exit_rule_id} 执行当前阶段的真实退出行动；"
                "普通发言、沉默或口头宣布进入下一阶段都不能完成阶段切换。"
            )
            state.record_structured_output_issue(
                "phase_exit_not_selected", retried=attempt < retries
            )
            if attempt >= retries:
                break
            continue
        quality_issues = inspect_dialogue_intent(state, agent, intent)
        if phase_exit_rule_id:
            quality_issues = [issue for issue in quality_issues if issue.code != "missing_response"]
        if intent.action_type not in {"speak", "pass", "observe"}:
            from .mechanics import patch_changes_state
            if patch_changes_state(state, resolution.patch.operations):
                quality_issues = [issue for issue in quality_issues if issue.code not in {"self_repetition", "parroting"}]
        if quality_issues:
            if state.model_requests:
                state.model_requests[-1]["quality_issue_codes"] = [issue.code for issue in quality_issues]
            rejection = (
                "Intent 通过规则校验，但未通过对话质量门：\n"
                f"{quality_retry_instruction(quality_issues)}\n"
                "保持人物目标和行动方向不变，只修正上述问题。"
            )
            should_retry = attempt < _single_retry_setting("quality_retries")
            state.record_dialogue_quality_issues(
                [issue.code for issue in quality_issues],
                retried=should_retry,
            )
            if should_retry or any(issue.hard for issue in quality_issues):
                if not should_retry:
                    break
                continue
        state.record_generation_success()
        return _message_from_resolution(state, agent, intent, resolution)

    phase = state.phase_specs.get(state.current_phase)
    allowed_actions = list(getattr(phase, "allowed_action_types", []) or [])
    if phase_exit_rule_id:
        exit_rule = next((rule for rule in state.rules if rule.id == phase_exit_rule_id), None)
        if exit_rule is not None:
            fallback_intent = Intent(
                actor=agent.name,
                action_type=exit_rule.action_type,
                rule_id=exit_rule.id,
                target=agent.name if exit_rule.target_scope == "self" else "",
                action="结束当前阶段，按规则推进到下一阶段。",
                speech="这轮先到这里，按规则进入下一阶段。",
            )
            fallback_resolution = active_resolver.resolve(state, fallback_intent)
            if fallback_resolution.accepted:
                state.record_structured_output_fallback()
                return _fallback_message(state, agent, fallback_intent, fallback_resolution)
    pending = agent.pending_intents[-1] if agent.pending_intents else None
    if pending and (not allowed_actions or "speak" in allowed_actions):
        fallback_intent = safe_obligation_fallback(agent, pending, rejection)
        fallback_resolution = active_resolver.resolve(state, fallback_intent)
        if fallback_resolution.accepted and fallback_intent.reply_to_event_id:
            state.record_structured_output_fallback()
            return _fallback_message(
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
            state.record_structured_output_fallback()
            return _fallback_message(
                state,
                agent,
                fallback_intent,
                fallback_resolution,
            )
    state.record_generation_failure()
    if state.run_status == "blocked" and rejection:
        state.end_reason += f" 最近一次校验原因：{rejection[:240]}"
    return None


def _fallback_message(state, agent, intent, resolution):
    # Narrator inserts must not hide a sequence of failed actor generations.
    count = 1
    for previous in reversed(state.history):
        if previous.kind == "intervention":
            break
        if previous.speaker not in state.agents:
            continue
        if not previous.intent.get("generation_fallback"):
            break
        count += 1
    limit = config_int("simulation", "consecutive_fallback_limit", 3, minimum=1, maximum=6)
    state.failed_generation_count = count
    window_size = config_int("simulation", "fallback_window_size", 12, minimum=4, maximum=30)
    window_limit = config_int("simulation", "fallback_window_limit", 6, minimum=2, maximum=20)
    previous_actors = [m for m in state.history if m.speaker in state.agents][-(window_size - 1):]
    recent_fallbacks = 1 + sum(bool(m.intent.get("generation_fallback")) for m in previous_actors)
    if count >= limit or recent_fallbacks >= window_limit:
        state.run_status = "blocked"
        state.end_kind = "blocked"
        recent_rule_rejection = next(
            (entry.get("rule_rejection_reason") for entry in reversed(state.model_requests[-8:])
             if entry.get("rule_rejection_reason")), None,
        )
        cause = (f"最近一次规则拒绝：{recent_rule_rejection}。" if recent_rule_rejection else
                 "请检查生成格式、对话质量校验与当前阶段规则。")
        state.end_reason = (
            f"角色行动连续落入兜底（连续 {count} 次，最近 {len(previous_actors) + 1} 次中 "
            f"{recent_fallbacks} 次），已暂停以避免无效推演。{cause}"
            "请从暂停前检查点创建分支继续，或修正设定重新生成。"
        )
    message = _message_from_resolution(state, agent, intent, resolution)
    message.intent["generation_fallback"] = True
    return message


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


@bounded_operation("simulation")
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
    factions = {
        agent.faction.strip() for agent in state.agents.values()
        if agent.faction.strip()
    }
    reveal_phase = any(
        marker in state.current_phase.lower()
        for marker in ("reveal", "result", "settlement", "揭晓", "公布", "结算", "复盘")
    )
    protect_hidden_competition = (
        len(factions) > 1
        and not reveal_phase
        and state.arc_state.progress < 0.65
        and not (
            getattr(state.world_spec, "audience_policy", "limited") == "omniscient"
            and getattr(state.world_spec, "reveal_policy", "preserve_suspense") == "allow_reveal"
        )
    )
    visibility = forced_visibility or (
        "public"
        if protect_hidden_competition or state.narration_count % 2 == 0
        else "audience_only"
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
    retries = max(
        _single_retry_setting("parse_retries"),
        _single_retry_setting("quality_retries"),
    )
    rejection = ""
    for attempt in range(retries + 1):
        attempt_prompt = prompt if not rejection else (
            prompt + "\n\n【上一次旁白被拒绝】\n" + rejection
            + "\n只输出修正后的完整 JSON 对象。"
        )
        response = complete(state, active_llm,
            attempt_prompt,
            purpose="narration",
            max_tokens=_token_budget("narration_max_tokens", 480, 360, 900),
        )
        parsed = parse_narrator_event(response.text)
        if parsed and parsed.get("skip"):
            phase = state.phase_specs.get(state.current_phase)
            mandatory = (bool(getattr(phase, "event_only", False))
                         or getattr(phase, "advance_when", "") == "after_event"
                         or any(i.status == "pending" for i in active_guidance(state)))
            if not mandatory:
                return None
            parsed = None
        if parsed is None:
            finish_reason = str(getattr(response, "finish_reason", "") or "")
            truncated = finish_reason == "length" or (
                str(response.text or "").lstrip().startswith("{")
                and not str(response.text or "").rstrip().endswith("}")
            )
            code = "narration_json_truncated" if truncated else "narration_json_invalid"
            should_retry = attempt < _single_retry_setting("parse_retries")
            state.record_structured_output_issue(code, retried=should_retry)
            rejection = "旁白 JSON 被截断，请缩短 narration 并闭合 JSON。" if truncated else (
                "旁白不是合法 JSON，请严格按 schema 输出。"
            )
            if not should_retry:
                break
            continue
        # The caller owns visibility; a model must not validate against an
        # audience-only mode and then publish the result to characters.
        parsed["visibility"] = visibility
        quality_issues = inspect_narration_event(
            state,
            parsed["narration"],
            visibility=visibility,
        )
        from .narrative_grounding import inspect_claims
        quality_issues.extend(inspect_claims(state, parsed))
        if quality_issues:
            if state.model_requests:
                state.model_requests[-1]["quality_issue_codes"] = [issue.code for issue in quality_issues]
            should_retry = attempt < _single_retry_setting("quality_retries")
            state.record_narration_quality_issues(
                [issue.code for issue in quality_issues],
                retried=should_retry,
            )
            rejection = quality_retry_instruction(quality_issues)
            if should_retry or any(issue.hard for issue in quality_issues):
                parsed = None
                if not should_retry:
                    break
                continue
        break
    if parsed is None:
        state.record_generation_failure()
        return None
    verified = {}
    if getattr(state.world_spec, "execution_version", 1) == 2:
        from .beat_evidence import verify_narrative_candidates
        verified = verify_narrative_candidates(state, active_llm, parsed["resolved_beat_ids"])
    resolved_beat_ids = validate_resolved_beats(state, parsed["resolved_beat_ids"], verified)
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
    intent_payload["narration_phase"] = state.current_phase
    intent_payload["evidence_event_ids"] = parsed.get("evidence_event_ids", [])
    intent_payload["arc_updates"] = {
        "resolved_beat_ids": resolved_beat_ids,
        "verified_beats": verified,
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


@bounded_operation("simulation")
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
    factions = {a.faction for a in state.agents.values() if a.faction}
    def reversed_winner(rule):
        return (rule.kind == "faction_parity" and rule.winner in factions and rule.winner != rule.faction
                or any(reversed_winner(child) for child in rule.conditions))
    if any(reversed_winner(rule) for rule in state.termination_rules):
        state.run_status = "blocked"
        state.end_kind = "blocked"
        state.end_reason = "胜负规则阵营方向与获胜方冲突，需重新生成场景。"
        raise SceneChatError("invalid_termination_contract",
                             "该存档的胜负规则阵营方向与获胜方冲突。请重新生成场景；不会继续推演或自动改写既有剧情。",
                             stage="simulation", status_code=409)
    state.evaluate_termination()
    if state.ended:
        return None
    scheduler_index = state._scheduler_index
    decision = (scheduler or SimulationScheduler()).decide(state)
    if decision.kind == "blocked":
        state.run_status = "blocked"
        state.end_kind = "blocked"
        state.end_reason = decision.reason
        raise SceneChatError(
            "simulation_scheduler_blocked",
            decision.reason + "。请检查场景阶段与角色规则后重新生成；本轮未调用模型。",
            stage="simulation",
            status_code=409,
        )
    guidance_waiting = any(item.status == "pending" for item in active_guidance(state))
    should_narrate = decision.kind == "agent" and (
        guidance_waiting or should_insert_narration(state)
    )
    if should_narrate:
        narration = simulate_narration(
            state,
            knowledge_base,
            llm=llm,
            resolver=resolver,
            forced_visibility="public" if guidance_waiting else None,
        )
        if narration is not None:
            # Inserting a narrator must not consume the selected actor's turn.
            state._scheduler_index = scheduler_index
            state.last_scheduler_decision = {
                "kind": "narration", "actor_name": "", "at_turn": state.turn_count,
                "reason": "应用导演引导" if guidance_waiting else "按节奏插入环境事件",
            }
            return narration
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
