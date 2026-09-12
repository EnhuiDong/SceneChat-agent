from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from Character import CHARACTER_SYSTEM_PROMPT
from World import WORLD_SYSTEM_PROMPT
from .config import config_int
from .build_control import BuildControl, CURRENT_BUILD
from .scenario_patch import apply_scenario_patch
from .errors import SceneChatError
from .recovery import fingerprint, is_transient, retry_delay, repair_local_fields

from .providers import get_generation_chat_model
from .scenario import (
    CharacterSpec,
    ScenarioBrief,
    ScenarioPackage,
    ScenarioValidationError,
    WorldSpec,
    extract_json_object,
    normalize_scenario_phase_references,
    validate_scenario_package,
)


BRIEF_SYSTEM_PROMPT = """你是 SceneChat 的用户约束分析器。你的任务是把用户输入整理成结构化约束账本，而不是创作故事正文。

工作规则：
1. 用户明确给出的题材、人物、人数、身份、技能、关系、规则、剧情节点、结局、场地和风格都是 locked 硬约束。
2. 短输入要补足正常运行所需的默认信息；详细输入要尽量少添加，只补真正缺失的运行条件。
3. 将用户已规定必须发生的内容放入 fixed_canon；将希望发生但实现方式可由角色推演的内容放入 target_beats。
4. 识别信息可见性：公开信息用 public，导演秘密用 director_only，指定角色或身份可见时用 agent:<名称> 或 role:<身份>。
5. 对轻微歧义选择最符合上下文的解释并写入 assumptions。互相冲突的硬约束写入 contradictions，并采用“更具体、明确列举、位置更靠后的要求优先”的可重复规则解决。
6. requested_character_count 必须是本次实际应生成的人数。用户未指定时，根据题材合理选择；单人场景允许 1 人。
7. 不得从单个概念词擅自推导整体时代或审美。“AI 玩家”不等于赛博朋克；没有明确风格要求时，采用最少额外假设的现实实现。
8. 用户未命名时，只记录需要生成独立姓名，不要在 assumptions 预先安排“林探长/苏医生”等职业姓名模板。公开的阵营人数与具体谁属于该阵营是不同事实，不能因为身份隐藏就把用户明确的游戏构成一并当作秘密。

只输出一个 JSON 对象，不要使用 Markdown 代码块：
{
  "input_mode": "short|partial|detailed",
  "genre": "题材",
  "premise": "核心设想",
  "requested_character_count": 角色总数,
  "requested_opening_scene": "用户明确指定的开场；未指定则为空字符串",
  "constraints": [
    {
      "id": "C001",
      "category": "cast|rule|world|relationship|ability|plot|ending|style|visibility|other",
      "content": "一条原子化约束",
      "source_excerpt": "对应的用户原文",
      "visibility": "public|director_only|agent:名称|role:身份",
      "locked": true
    }
  ],
  "assumptions": ["系统补全或歧义解释"],
  "contradictions": ["冲突以及采用的解释"],
  "fixed_canon": ["必须发生或保持不变的内容"],
  "target_beats": ["应努力引导但不强迫角色具体选择的节点"]
}"""


WORLD_JSON_INSTRUCTION = """在遵守上方所有世界设计规则的同时，只输出一个 JSON 对象，不要使用 Markdown 代码块：
{
  "title": "贴合题材的名称",
  "opening_scene": "可直接开始推演的一段具体开场，不得使用与用户题材无关的默认地点",
  "public_world_markdown": "只包含所有在场角色都可知道的公共环境、规则和背景的 Markdown。严禁写入隐藏身份、秘密任务、仅某些身份知道的机制或导演计划",
  "director_notes_markdown": "只供导演使用的隐藏机制、秘密真相、角色差异化信息规则和剧情约束 Markdown",
  "public_rules": ["必须持续执行的公共规则"],
  "execution_version": 2,
  "audience_policy": "limited",
  "reveal_policy": "preserve_suspense",
  "entities": {},
  "phases": ["若题材有回合/昼夜/会议等阶段，按顺序列出；自由场景可只写自由推进"],
  "initial_state": {"公共状态变量名": "初始值"},
  "locations": ["推演中允许存在的地点"],
  "relationship_dimensions": [
    {"id":"cooperation","label":"合作倾向","low_label":"对抗","high_label":"协作","description":"本场景中这一关系维度的含义"}
  ],
  "facts": [
    {"id":"fact-1", "content":"一条原子事实", "visibility":["public|director_only|audience_only|agent:姓名|role:身份|location:地点"], "source":"user|world|inferred", "protected_propositions":["非公开事实的 1–3 条不可被未知角色直接断言的核心命题；公开事实可为空"], "covered_constraint_ids":["对应约束 ID"]}
  ],
  "state_schema": {
    "状态变量名": {"value_type":"string|integer|boolean|enum|any", "visibility":["public"], "mutable_by":["resolver"], "allowed_values":[]}
  },
  "scheduler": "round_robin|phase_order|initiative|urgency_director|event_first",
  "phase_specs": [
    {"name":"阶段名", "scheduler":"phase_order", "allowed_action_types":["speak","vote","use_ability"], "actor_roles":[], "advance_when":"all_eligible_acted|all_active_voted|after_event|manual", "next_phase":"下一阶段", "event_only":false}
  ],
  "rules": [
    {"id":"rule-1", "description":"可执行规则", "action_type":"vote", "phases":["表决"], "allowed_roles":[], "target_scope":"proposal", "priority":0, "effect_mode":"rule", "behavior_template":"", "effects":[{"op":"record_vote", "target":"$target"}], "visibility":["public"]}
  ],
  "termination_conditions": ["自然结束或胜负条件"],
  "termination_rules": [
    {"id":"end-1", "kind":"faction_eliminated|faction_parity|world_equals|entity_equals|all_goals_completed|all_active_at_location|all_of|any_of|manual", "description":"结束说明", "phases":["允许判定胜负的结算/公布阶段；自由场景可为空"], "faction":"可选阵营", "opposing_factions":[], "target":"entity_equals 的实体 ID", "key":"相等比较使用的状态键", "value":"目标值，必须保持实际类型", "location":"all_active_at_location 使用的地点", "winner":"胜方", "conditions":[]}
  ],
  "fixed_canon": ["从用户约束账本继承的必须发生内容"],
  "target_beats": ["需要跟踪但不强迫具体角色选择的节点"],
  "beat_specs": [
    {"id":"beat-1","description":"可观察、可判定是否已经发生的剧情节点","required":false,"weight":1,"prerequisites":[],"phase_hint":"可选阶段名","resolution_signals":["足以判定节点完成的具体现象"]}
  ],
  "covered_constraint_ids": ["本世界设定已落实的约束 ID"]
}

多个条件必须同时成立时，使用一个 kind=all_of 的顶层规则，把原子规则放入 conditions；任选其一时使用 any_of。不得把“且/全部/同时”条件拆成多个并列顶层规则，因为顶层规则之间按任一满足处理。

新场景使用 execution_version=2：行动名称不隐含效果，必须显式填写 effects。rule.effect_mode=rule（默认，仅规则结果）、ability（仅能力结果）、stack（明确叠加）。能力使用次数与 consume_resource 成本始终校验扣除，不要重复声明 consume_ability。规则重叠用不同 priority 选最大者，禁止依赖顺序。没有规则时使用能力自身效果。会议 vote 只记录提案投票，不淘汰角色；医治 heal 不等于复活；inspect 只返回规则准许的事实。只有用户设定确实要求传统桌游效果时，才可显式选 behavior_template=legacy_tabletop，此时 effects 留空；不得根据题材名称强制套模板。
target_scope 除角色 same_location/any_active/character/self/none，还支持 location/object/proposal。物品或提案声明在 entities 对象中，例如 {"proposal-A":{"kind":"proposal","state":{"passed":false}}}；通过 set_entity(target,key,value) 修改已声明的同类型字段。地点用 locations。资源在角色 resources 声明。move_agent 的 target=$actor、value=$target；record_vote 的 target=$target。不能发明运算符或运行代码；复杂统计若无法由已有操作精确表达，不能伪称支持。
观众策略 audience_policy=limited|omniscient、reveal_policy=preserve_suspense|allow_reveal 应遵从用户要求，默认 limited + preserve_suspense。全知读者镜头仍是 audience_only，不能因此让角色知晓秘密。

每个非 event_only 阶段的 allowed_action_types 必须至少包含 pass、observe、speak、act 之一作为安全兜底，即使该阶段主要执行投票、查验或自定义行动；兜底行动用于模型连续提交非法 Intent 时保持阶段可推进，不能省略。
advance_when=manual 表示阶段可无限持续；如果同时填写 next_phase，必须提供一条在本阶段可执行且包含 set_phase 到 next_phase 的规则，不能只在自然语言中说“之后进入下一阶段”。阵营对抗或比赛场景的 termination_rules 必须明确 winner，并能区分主要胜负结果。
有夜间技能、治疗、反制或结算顺序时，termination rule 必须用 phases 限制到结算/公布阶段，不能在中间行动后提前判胜。

状态 effect 仅允许 set_world、increment_world、set_entity、move_agent、set_agent_status、set_resource、consume_resource、set_goal_status、set_relationship、record_vote、clear_votes、set_phase、add_known_fact、protect_agent、clear_protections。模板中的 $actor、$target、$value 会在运行时由 Resolver 安全替换。
提案投票还可使用 settle_votes：target=提案 ID，key=该提案的布尔状态字段，amount=所需同意票数。在全部可行动角色投票后按该提案得票数写 true/false；须放在 record_vote 后、clear_votes 或 set_phase 前。没有全员投票时不结算。不支持表达式、条件脚本或任意统计。世界环境变量若允许旁白更新，state_schema.mutable_by 必须包含 director；关键目标结果只允许 resolver，避免旁白越权宣布结果。
实体结果作为结束条件时使用 termination_rules.kind=entity_equals，并填写 target=实体 ID、key=状态键、value=准确类型的目标值；可以放进 all_of/any_of。

beat_specs 必须与 target_beats 一一对应并使用稳定 ID。用户明确要求必须发生的节点标 required=true；普通期望节点不得标成硬约束。description 和 resolution_signals 必须描述可观察结果，不能要求某角色违背自主判断作出指定选择；prerequisites 只引用前面已声明的 beat ID。
每个节点声明 completion_mode=state|narrative。可执行结果必须用 state 和 completion_conditions（所有条件同时满足）：world_equals(key,value)、entity_equals(target,key,value)、agent_location(target,value)、goal_equals(target,key,value)，每个条件使用 kind 字段。状态键必须声明，目标必须存在。找到线索不等于找到失踪者，治疗步骤不等于治愈。承诺、和解等使用 narrative，写清 resolution_signals，之后由已提交角色事件证据核验；旁白声称完成不是证据。开放场景无需虚构固定终点或节点。

特别注意：公共世界会直接成为所有角色的知识。任何并非所有角色都知道的信息只能进入 director_notes_markdown 或带精确非 public scope 的 facts。桌游、审判、比赛等规则题材必须给出可执行 phase_specs、rules 与 termination_rules；自由谈话可以保持精简。天亮公布、结算、广播等没有角色行动的阶段必须设置 scheduler=event_first、advance_when=after_event、event_only=true，并给出 next_phase；唯一例外是有适用结构化结束规则的终局阶段，可以不设 next_phase。

relationship_dimensions 应选择 2–5 个对本场景真正有用、含义互不重复的主观关系维度，必须兼容题材而不是默认套用猜疑游戏：战斗可使用敌对/协作、威胁判断、敬重或服从；职场可使用合作、可靠判断、影响力；家庭或情感场景可使用亲密、信赖、依赖或边界感。不要把战斗胜负、生命值、距离等客观状态伪装成关系维度。若题材没有特殊需要，可使用 cooperation、confidence、regard 三个通用维度。"""

RUNTIME_GENERATION_GUIDANCE = """
set_phase 使用 value 存放目标阶段原文，例如 {"op":"set_phase","value":"秘密投票"}；target 不是目标阶段字段。不得写 value=null 再把阶段填入 target。
结构化协议必须逐字段遵循：effects 是数组，每项操作的 op/key/target/value/amount 均在同一层，禁止嵌套 params、arguments、path 或自造操作（例如 check_condition_and_set）。set_agent_status 只允许 key=alive/active 和布尔 value；其他人物数值放 resources。实体结构必须是 {"实体ID":{"kind":"object或proposal","state":{"状态键":初始值}}}，不得把 state 改叫 properties。地点在 locations、opening_scene 与人物 initial_location 中使用一致的原文名称，不要混用英文 ID 和中文名。
V2 提案表决的最小操作示例（仅在用户要求此机制时使用，不能强加给其他场景）：先声明 entities={"proposal-A":{"kind":"proposal","state":{"passed":false}}}，对应 rule.action_type=vote、target_scope=proposal、effect_mode=rule、effects=[{"op":"record_vote","target":"$target"},{"op":"settle_votes","target":"$target","key":"passed","amount":2}]。两人投票通过的结束规则为 {"id":"end","kind":"entity_equals","target":"proposal-A","key":"passed","value":true,"description":"提案通过"}。不得增加未经支持的条件脚本或把 passed 当成角色状态。
运行可达性与人物命名约束：
- actor_roles/allowed_roles 为空数组表示所有可行动角色；非空时必须与实际 CharacterSpec.role 一致，禁止虚构 all_active 职业或名单外的 host。规则 action_type 必须在其阶段 allowed_action_types 中。
- 主持流程由公开环境事件或已有角色完成，不能为了主持额外增加用户未要求的角色。有限回合的讨论优先 all_eligible_acted；只有确实允许无限讨论时才用 manual。manual 的退出规则必须有真实有权执行的角色和被允许的动作。
- 普通环境事件阶段使用 after_event 并明确 next_phase；事件阶段链必须能回到角色行动阶段。终局可以没有 next_phase，但必须有在该阶段适用的结构化结束规则；不得依赖旁白永远循环。
- world_equals 是类型严格的相等比较，不执行 >=、<= 或表达式；存活阵营数量的胜负使用 faction_eliminated/faction_parity，不能依赖无人维护的计数字段。
- 普通问答或质疑使用 speak + question/challenge，inspect 仅用于场景明确授权的事实查验能力，不得让普通玩家提问就读取秘密阵营。
- 用户未指定姓名时为每人生成符合时代文化的独立姓名，职业只写在身份字段；不得用“姓氏+职业”代替姓名。用户明确给出的姓名、称呼、编号或代号必须保留。导演笔记若出现角色信息必须与角色表一致，不能把身份写成“示例、稍后随机、根据表现动态分配”。
- 只有用户明确要求必经的节点才标 required=true，系统补全的剧情目标不得自动升级成硬约束；节点完成必须有实际事件支持，灯光变化不能代替规则宣读或角色讨论。
- 新世界 execution_version=2 时，角色能力必须显式写 effects；行动名称本身不会查验身份、复活或杀人。若配套规则选择 rule 来源，能力效果仅扣 consume_resource 成本；能力次数由引擎扣除。目标只可使用声明的 target_scope，效果值必须符合世界 schema 和角色 resources 类型。旧版示例不是隐式授权。
"""


CHARACTER_JSON_INSTRUCTION = """在遵守上方所有角色设计规则的同时，只输出一个 JSON 对象，不要使用 Markdown 代码块：
{
  "characters": [
    {
      "id": "character-1",
      "name": "姓名",
      "role": "运行阶段和规则使用的身份或职能",
      "faction": "阵营；无阵营时为空字符串",
      "initial_location": "必须来自世界 locations",
      "public_identity": "公开身份",
      "public_traits": "可观察的形象、表达和行为特征",
      "public_background": "其他角色合理知道的背景",
      "private_identity": "隐藏身份、秘密、底牌和私有任务",
      "personality": "性格结构及成因",
      "goals": ["目标、顾虑和可接受代价"],
      "core_beliefs": ["只有当用户设定或人物经历确实支持时，填写会持续影响选择的价值信念；否则为空数组"],
      "knowledge": {
        "确定知道": "...",
        "不知道": "...",
        "怀疑": "...",
        "可能错误相信": "..."
      },
      "decision_logic": "在本题材关键局面中的观察、推理、表达和行动逻辑",
      "voice_profile": {
        "register": "自然口语、正式、古典、粗粝、克制等；优先使用用户明确设定",
        "sentence_length": "短句为主|中等|长短交替",
        "directness": 0.5,
        "emotional_expressiveness": 0.5,
        "politeness": 0.5,
        "humor_style": "没有时为空字符串",
        "rhetorical_habits": ["稳定但不过度重复的表达策略"],
        "avoidances": ["该角色不会使用的措辞或表达方式"],
        "vocabulary_hints": ["符合身份与时代的少量自然词汇"]
      },
      "relationships": {"其他角色姓名": "该角色的主观认知，不得包含无权知道的秘密"},
      "relationship_facets": {"其他角色姓名": {"world.relationship_dimensions 中的维度 ID": 0.5}},
      "observation_value": "该角色的互动观察价值",
      "resources": {"资源名":"初始数量或状态"},
      "abilities": [
        {"id":"ability-1", "name":"能力名", "action_type":"inspect|protect|eliminate|heal|move|vote|act", "description":"能力与限制", "phases":["允许阶段"], "uses":1, "target_scope":"same_location|any_active|self|none", "visibility":["public|director_only|agent:姓名|role:身份"], "effects":[]}
      ],
      "known_fact_ids": ["该角色初始可知的 world facts ID；同时仍会由 visibility 自动授权"],
      "false_beliefs": ["该角色可能错误相信的内容"],
      "covered_constraint_ids": ["该角色落实的用户约束 ID"]
    }
  ]
}

角色数量必须与约束账本的 requested_character_count 完全一致。公开字段不能泄露 private_identity、faction、私有知识、隐藏阵营或秘密任务。能力必须结构化且引用真实阶段；普通交谈能力可以为空。夜袭、查验、守护等秘密技能默认 director_only；只有在场角色可直接观察的技能才标 public。inspect/protect/eliminate/heal/move/vote 已有通用 Resolver 行为，effects 可留空；只有额外题材状态变化才填写 effects。

voice_profile 只负责稳定语言倾向，不得用夸张口癖替代人物塑造。用户明确写过语气、时代用语、措辞禁忌或表达习惯时必须原样落实；用户未写时仅根据身份、年龄、时代和性格做克制推断。不同角色至少应在直接程度、情绪外显、礼貌程度或表达策略中的两项存在可解释差异。

core_beliefs 是可选的人物驱动力，不是必填装饰。只有用户明确给出，或人物经历足以支持一种会跨情境影响选择的价值信念时才填写；普通人物允许为空。relationship_facets 只能引用世界已经声明的 relationship_dimensions，数值 0–1 表示从 low_label 到 high_label 的位置；战斗人物也不能被强制套用亲密或猜疑维度。"""


def _content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def _repair_attempts(key: str, default: int) -> int:
    return config_int("scenario", key, default, minimum=0, maximum=2)


def scenario_semantic_repair_attempts() -> int:
    return _repair_attempts("semantic_repair_retries", 2)


def _is_transient_transport_error(exc: Exception) -> bool:
    return is_transient(exc)


def _invoke_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = 12000,
    allow_json_repair: bool = True,
    validate_payload=None,
) -> dict[str, Any]:
    if CURRENT_BUILD.get() is None:
        control = BuildControl()
        control.begin_step("scenario_generation")
        with control.activate():
            return _invoke_json(system_prompt, user_prompt, temperature, max_tokens, allow_json_repair, validate_payload)
    control = CURRENT_BUILD.get()
    key = fingerprint([system_prompt, user_prompt])
    ledger = control.recovery.setdefault("json", {})
    failures = ledger.setdefault(key, {"outputs": {}, "count": 0})
    def check_stalled():
        if failures["count"] >= 6 or max(failures["outputs"].values(), default=0) >= 2:
            raise SceneChatError("structured_output_stalled",
                                 "模型反复返回相同的无效结构，已停止自动重试。请调整模型或设定后重新生成；检查点已保留。",
                                 stage=control.stage, status_code=422)
    check_stalled()
    repairs = _repair_attempts("json_repair_retries", 2) if allow_json_repair else 0
    last_error: Exception | None = None
    request_limit = config_int("scenario", "max_requests_per_step", 4, minimum=1, maximum=8)
    for attempt in range(repairs + 1):
        control.check()
        correction = "" if attempt == 0 else (
            f"\n\n第 {attempt} 次自动修复：上一次响应未通过 JSON 或补丁结构校验。"
            f"具体问题：{str(last_error)[:1500]}。"
            "保留任务约束，省略不必要解释，严格生成一份完整且闭合的 JSON；不要复述失败输出。"
        )
        transport_retries = _repair_attempts("transport_retries", 1)
        for transport_attempt in range(transport_retries + 1):
            control.check()
            if control.step_requests >= request_limit:
                raise ScenarioValidationError(["当前生成步骤已达到请求次数预算，请从检查点重试"])
            control.step_requests += 1
            control.progress(request_attempt=control.step_requests, request_limit=request_limit,
                             json_repair_attempt=attempt, transport_retry=transport_attempt,
                             input_chars=len(system_prompt) + len(user_prompt), max_output_tokens=max_tokens)
            llm = get_generation_chat_model(
                temperature=temperature if attempt == 0 else 0,
                max_tokens=max_tokens,
                transport_retries=0,
            )
            try:
                from .telemetry import measured_call
                if not hasattr(control, "model_requests"):
                    control.model_requests = []
                response = measured_call(
                    control.model_requests, stage=control.stage,
                    purpose="json_repair" if attempt else "structured_generation",
                    model=getattr(llm, "model_name", "configured"),
                    operation=lambda: llm.invoke(
                        [("system", system_prompt + correction), ("user", user_prompt)],
                        response_format={"type": "json_object"},
                    ),
                )
                control.check()
                raw = getattr(response, "raw", None)
                usage = getattr(raw, "usage", None)
                control.progress(
                    finish_reason=getattr(response, "finish_reason", ""),
                    output_tokens=getattr(usage, "completion_tokens", None),
                )
                break
            except Exception as exc:
                if (
                    transport_attempt >= transport_retries
                    or not _is_transient_transport_error(exc)
                ):
                    raise
                control.progress(reason="模型请求超时或连接失败，准备进行一次受预算限制的重试")
                retry_delay(control, transport_attempt)
            finally:
                close = getattr(llm, "close", None)
                if close:
                    close()
        try:
            payload = extract_json_object(_content(response))
            if validate_payload:
                validate_payload(payload)
            ledger.pop(key, None)
            control.save_recovery()
            return payload
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            last_error = exc
            output_key = fingerprint(_content(response))
            failures["outputs"][output_key] = failures["outputs"].get(output_key, 0) + 1
            failures["count"] += 1
            control.save_recovery()
            check_stalled()
    assert last_error is not None
    raise last_error


def _brief_issues(brief: ScenarioBrief) -> list[str]:
    issues = []
    if not brief.premise:
        issues.append("约束分析缺少核心设想 premise")
    if brief.requested_character_count is None or brief.requested_character_count < 1:
        issues.append("约束分析没有给出有效角色数量")
    if not brief.constraints:
        issues.append("约束分析没有提取任何用户约束")
    return issues


def generate_scenario_brief(user_prompt: str, scene_override: str = "") -> ScenarioBrief:
    def validate_brief(candidate):
        issues = _brief_issues(ScenarioBrief.from_mapping(candidate))
        if issues:
            raise ScenarioValidationError(issues)
    payload = _invoke_json(
        BRIEF_SYSTEM_PROMPT,
        "【用户原始输入】\n"
        f"{user_prompt}\n\n"
        "【用户单独填写的初始场景覆盖项】\n"
        f"{scene_override or '未填写，由系统从题材中生成'}",
        temperature=0.2,
        max_tokens=4000,
        validate_payload=validate_brief,
    )
    brief = ScenarioBrief.from_mapping(payload)
    if scene_override.strip():
        brief.requested_opening_scene = scene_override.strip()
    issues = _brief_issues(brief)
    for attempt in range(scenario_semantic_repair_attempts()):
        if not issues:
            break
        repair_payload = _invoke_json(
            BRIEF_SYSTEM_PROMPT,
            "【用户原始输入】\n"
            f"{user_prompt}\n\n【需要修复的约束账本】\n"
            f"{json.dumps(brief.to_dict(), ensure_ascii=False, indent=2)}\n\n"
            "【确定性校验错误】\n"
            f"{json.dumps(issues, ensure_ascii=False)}\n\n"
            "只修复这些错误，不得丢失用户原文约束。",
            temperature=0.0,
            max_tokens=4000,
            validate_payload=validate_brief,
        )
        brief = ScenarioBrief.from_mapping(repair_payload)
        if scene_override.strip():
            brief.requested_opening_scene = scene_override.strip()
        issues = _brief_issues(brief)
    if issues:
        raise ScenarioValidationError(issues)
    return brief


def generate_world_spec(user_prompt: str, brief: ScenarioBrief) -> WorldSpec:
    payload = _invoke_json(
        f"{WORLD_SYSTEM_PROMPT}\n\n{WORLD_JSON_INSTRUCTION}\n\n{RUNTIME_GENERATION_GUIDANCE}",
        "【用户原始输入——最高约束】\n"
        f"{user_prompt}\n\n"
        "【结构化约束账本】\n"
        f"{json.dumps(brief.to_dict(), ensure_ascii=False, indent=2)}",
        temperature=0.7,
        max_tokens=10000,
        validate_payload=lambda candidate: WorldSpec.from_mapping({**candidate, "execution_version": 2}),
    )
    payload["execution_version"] = 2
    world = WorldSpec.from_mapping(payload)
    if brief.requested_opening_scene:
        world.opening_scene = brief.requested_opening_scene
    return world


def generate_character_specs(
    user_prompt: str,
    brief: ScenarioBrief,
    world: WorldSpec,
) -> list[CharacterSpec]:
    def validate_characters(candidate):
        characters = candidate.get("characters")
        if not isinstance(characters, list) or len(characters) != brief.requested_character_count:
            raise ScenarioValidationError([f"/characters 必须是包含 {brief.requested_character_count} 名角色的数组"])
        for index, item in enumerate(characters, 1):
            if not isinstance(item, dict):
                raise ScenarioValidationError([f"/characters/{index - 1} 必须是角色对象"])
            CharacterSpec.from_mapping(item, index)
    payload = _invoke_json(
        f"{CHARACTER_SYSTEM_PROMPT}\n\n{CHARACTER_JSON_INSTRUCTION}\n\n{RUNTIME_GENERATION_GUIDANCE}",
        "【用户原始输入——所有明确细节均为最高级约束】\n"
        f"{user_prompt}\n\n"
        "【结构化约束账本】\n"
        f"{json.dumps(brief.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        "【结构化世界设定——导演信息只用于分配正确的私有档案】\n"
        f"{json.dumps(asdict(world), ensure_ascii=False, indent=2)}",
        temperature=0.75,
        max_tokens=16000,
        validate_payload=validate_characters,
    )
    raw_characters = payload.get("characters")
    if not isinstance(raw_characters, list):
        raise ScenarioValidationError(["角色生成响应缺少 characters 数组"])
    return [
        CharacterSpec.from_mapping(item, index)
        for index, item in enumerate(raw_characters, start=1)
        if isinstance(item, dict)
    ]


def repair_scenario_package(
    user_prompt: str,
    package: ScenarioPackage,
    issues: list[str],
) -> ScenarioPackage:
    repair_prompt = """你是 SceneChat 结构化设定修复器。不得修改用户约束账本。只修复校验错误，输出小型字段补丁 JSON，不要重写完整世界或角色：
{
  "changes": [{"path":"/world/phase_specs/0/actor_roles", "value":[]}]
}
path 使用 JSON Pointer，数字为从 0 开始的数组索引。仅可修改 /world/<字段> 或 /characters/<索引>/<字段> 下的值。禁止修改 brief、角色数量及角色 id，禁止替换整个 world 或整个人物。只输出发生变化的字段，保持其他字段原样；新增字典键的父路径必须已存在。修改姓名必须同时修复所有关联引用。最多 80 个不重叠的修改；不要添加解释或未变化字段。
优先最小修复：例如 actor_roles 引用了不存在的角色，只修复 actor_roles；不要同时新增胜负规则、角色能力、剧情或世界状态。以下通用约束只限制确实需要修改的字段，不要求补全原设定中所有可选字段。未被报错涉及且无关联依赖的字段必须保持不变。
所有 covered_constraint_ids 必须真实对应其落实位置。公共世界仍不得包含导演秘密。
如果错误涉及规则运行时，只修改报错涉及的 phase_specs、rules、state_schema、entities 或 termination_rules，以及必要关联。termination_rules.kind 只支持 faction_eliminated、faction_parity、world_equals、entity_equals、all_goals_completed、all_active_at_location、all_of、any_of、manual；禁止创造 ai_win、human_win、entity_state_equals 等新 kind。all_of/any_of 必须有非空 conditions。角色阶段保留 pass/observe/speak/act 至少一种兜底；manual 且有 next_phase 时必须有可执行 set_phase 规则。
严格保留 world.execution_version。版本1才有 move/vote/inspect/protect/eliminate/poison/heal 的隐式内置效果。版本2行动名称没有隐式结果，必须填写白名单 effects；effect_mode=rule/ability/stack 控制来源，不得为了修复把正常 effects 清空。legacy_tabletop 仅可显式用于用户确实要求的传统桌游机制，不能套到会议、医疗、战斗等场景。consume_ability 由引擎处理，不得重复填写。禁止脚本、条件表达式和未支持操作，不要退回自然语言规则冒充可执行字段。"""
    control = CURRENT_BUILD.get()
    history = (control.recovery.get("semantic", {}).get("history", [])[-3:] if control else [])
    payload = _invoke_json(
        repair_prompt + "\n\n" + RUNTIME_GENERATION_GUIDANCE,
        "【用户原始输入】\n"
        f"{user_prompt}\n\n"
        "【不可修改的约束账本】\n"
        f"{json.dumps(package.brief.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        "【校验错误】\n"
        f"{json.dumps(issues, ensure_ascii=False, indent=2)}\n\n"
        "【先前修复结果（不要重复无进展方案）】\n"
        f"{json.dumps(history, ensure_ascii=False)}\n\n"
        "【待修复结果】\n"
        f"{json.dumps(package.to_dict(), ensure_ascii=False, indent=2)}",
        temperature=0.2,
        max_tokens=config_int("scenario", "repair_max_tokens", 4096, minimum=512, maximum=10000),
        validate_payload=lambda candidate: apply_scenario_patch(package, candidate),
    )
    result = apply_scenario_patch(package, payload)
    control = CURRENT_BUILD.get()
    if control:
        control.progress(patched_paths=[item["path"] for item in payload["changes"]])
    return result


def validate_and_repair_package(user_prompt, package, *, save=None, repair=None):
    """Keep the best checkpoint; semantic retries must improve validation."""
    if CURRENT_BUILD.get() is None:
        with BuildControl().activate():
            return validate_and_repair_package(user_prompt, package, save=save, repair=repair)
    control = CURRENT_BUILD.get()
    save = save or (lambda value: None)
    repair = repair or repair_scenario_package
    normalize_scenario_phase_references(package)
    package, changes = repair_local_fields(package)
    save(package)
    issues = validate_scenario_package(package, user_prompt=user_prompt)
    ledger = control.recovery.setdefault("semantic", {"attempts": 0, "no_progress": 0, "history": []})
    attempts = 0
    if changes:
        control.progress(reason="已本地纠正明确的字段错位，无需请求模型",
                         patched_paths=[item["path"] for item in changes])
    def check_stalled():
        if ledger["no_progress"] >= 2 or ledger["attempts"] >= config_int(
                "scenario", "semantic_total_attempts", 6, minimum=1, maximum=12):
            raise SceneChatError("scenario_repair_stalled",
                                 "一致性修复未取得进展或已达累计上限，已保留较好的检查点。请修改设定或模型后重新生成；重复继续不会再调用修复模型。",
                                 stage="validation", status_code=422)
    for _ in range(scenario_semantic_repair_attempts()):
        if not issues:
            break
        check_stalled()
        control.progress(repair_attempt=ledger["attempts"] + 1, issue_count=len(issues), issues=issues[:30],
                         reason="按具体校验错误进行局部修复，并比较修复前后结果")
        # A transport failure does not count as a semantic result.
        candidate = repair(user_prompt, package, issues)
        normalize_scenario_phase_references(candidate)
        candidate, _ = repair_local_fields(candidate)
        candidate_issues = validate_scenario_package(candidate, user_prompt=user_prompt)
        improved = set(candidate_issues) < set(issues) and fingerprint(candidate.to_dict()) != fingerprint(package.to_dict())
        ledger["attempts"] += 1
        attempts += 1
        ledger["no_progress"] = 0 if improved else ledger["no_progress"] + 1
        ledger["history"].append({"before_issues": issues[:30], "after_issues": candidate_issues[:30],
                                  "accepted": improved,
                                  "patched_paths": control.details.get("patched_paths", [])})
        ledger["history"] = ledger["history"][-6:]
        if improved:
            package, issues = candidate, candidate_issues
            save(package)
        control.save_recovery()
        if issues:
            check_stalled()
    if issues:
        check_stalled()
        raise ScenarioValidationError(issues)
    ledger["no_progress"] = 0
    control.save_recovery()
    return package, attempts + bool(changes)


def generate_scenario_package(user_prompt: str, scene_override: str = "") -> ScenarioPackage:
    if CURRENT_BUILD.get() is None:
        with BuildControl().activate():
            return generate_scenario_package(user_prompt, scene_override)
    control = CURRENT_BUILD.get()
    control.begin_step("brief")
    brief = generate_scenario_brief(user_prompt, scene_override)
    control.begin_step("world")
    world = generate_world_spec(user_prompt, brief)
    control.begin_step("characters")
    characters = generate_character_specs(user_prompt, brief, world)
    control.begin_step("validation")
    package = ScenarioPackage(brief=brief, world=world, characters=characters)
    return validate_and_repair_package(user_prompt, package)[0]
