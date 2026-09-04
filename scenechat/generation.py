from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from Character import CHARACTER_SYSTEM_PROMPT
from World import WORLD_SYSTEM_PROMPT
from .config import config_int

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
    {"id":"rule-1", "description":"可执行规则", "action_type":"vote", "phases":["白天投票"], "allowed_roles":[], "target_scope":"same_location|any_active|self|none", "effects":[{"op":"record_vote", "target":"$target"}], "visibility":["public"]}
  ],
  "termination_conditions": ["自然结束或胜负条件"],
  "termination_rules": [
    {"id":"end-1", "kind":"faction_eliminated|faction_parity|world_equals|all_goals_completed|all_active_at_location|all_of|any_of|manual", "description":"结束说明", "phases":["允许判定胜负的结算/公布阶段；自由场景可为空"], "faction":"可选阵营", "opposing_factions":[], "key":"world_equals 使用的状态键", "value":"目标值", "location":"all_active_at_location 使用的地点", "winner":"胜方", "conditions":[]}
  ],
  "fixed_canon": ["从用户约束账本继承的必须发生内容"],
  "target_beats": ["需要跟踪但不强迫具体角色选择的节点"],
  "beat_specs": [
    {"id":"beat-1","description":"可观察、可判定是否已经发生的剧情节点","required":false,"weight":1,"prerequisites":[],"phase_hint":"可选阶段名","resolution_signals":["足以判定节点完成的具体现象"]}
  ],
  "covered_constraint_ids": ["本世界设定已落实的约束 ID"]
}

多个条件必须同时成立时，使用一个 kind=all_of 的顶层规则，把原子规则放入 conditions；任选其一时使用 any_of。不得把“且/全部/同时”条件拆成多个并列顶层规则，因为顶层规则之间按任一满足处理。

运行时已经内置 move、vote、inspect、protect、eliminate、poison、heal 的标准效果和能力次数扣减。此类 rule 的 effects 应留空，不能重复填写淘汰、查验、保护、投票、移动或 consume_ability；只有题材额外要求的世界状态、资源、目标或关系变化才写入 effects。普通 speak、observe、pass、act 规则也不得借 effects 越权执行专用行动。

每个非 event_only 阶段的 allowed_action_types 必须至少包含 pass、observe、speak、act 之一作为安全兜底，即使该阶段主要执行投票、查验或自定义行动；兜底行动用于模型连续提交非法 Intent 时保持阶段可推进，不能省略。
advance_when=manual 表示阶段可无限持续；如果同时填写 next_phase，必须提供一条在本阶段可执行且包含 set_phase 到 next_phase 的规则，不能只在自然语言中说“之后进入下一阶段”。阵营对抗或比赛场景的 termination_rules 必须明确 winner，并能区分主要胜负结果。
有夜间技能、治疗、反制或结算顺序时，termination rule 必须用 phases 限制到结算/公布阶段，不能在中间行动后提前判胜。

状态 effect 仅允许 set_world、increment_world、move_agent、set_agent_status、set_resource、consume_resource、set_goal_status、set_relationship、record_vote、clear_votes、set_phase、add_known_fact、protect_agent、clear_protections。模板中的 $actor、$target、$value 会在运行时由 Resolver 安全替换。

beat_specs 必须与 target_beats 一一对应并使用稳定 ID。用户明确要求必须发生的节点标 required=true；普通期望节点不得标成硬约束。description 和 resolution_signals 必须描述可观察结果，不能要求某角色违背自主判断作出指定选择；prerequisites 只引用前面已声明的 beat ID。

特别注意：公共世界会直接成为所有角色的知识。任何并非所有角色都知道的信息只能进入 director_notes_markdown 或带精确非 public scope 的 facts。桌游、审判、比赛等规则题材必须给出可执行 phase_specs、rules 与 termination_rules；自由谈话可以保持精简。天亮公布、结算、广播等没有角色行动的阶段必须设置 scheduler=event_first、advance_when=after_event、event_only=true，并给出 next_phase。

relationship_dimensions 应选择 2–5 个对本场景真正有用、含义互不重复的主观关系维度，必须兼容题材而不是默认套用猜疑游戏：战斗可使用敌对/协作、威胁判断、敬重或服从；职场可使用合作、可靠判断、影响力；家庭或情感场景可使用亲密、信赖、依赖或边界感。不要把战斗胜负、生命值、距离等客观状态伪装成关系维度。若题材没有特殊需要，可使用 cooperation、confidence、regard 三个通用维度。"""


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
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    terminal_markers = (
        "authentication", "unauthorized", "invalid api key", "invalid_api_key",
        "insufficient", "quota exhausted", "model not found", "invalid model",
    )
    if any(marker in text for marker in terminal_markers):
        return False
    return (
        name in {"apitimeouterror", "apiconnectionerror", "internalservererror"}
        or any(marker in text for marker in (
            "timed out", "timeout", "connection reset", "connection error",
            "temporarily unavailable", "bad gateway", "service unavailable",
            "gateway timeout", "502", "503", "504",
        ))
    )


def _invoke_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = 12000,
    allow_json_repair: bool = True,
) -> dict[str, Any]:
    repairs = _repair_attempts("json_repair_retries", 2) if allow_json_repair else 0
    last_error: Exception | None = None
    for attempt in range(repairs + 1):
        llm = get_generation_chat_model(
            temperature=temperature if attempt == 0 else 0,
            max_tokens=max_tokens,
        )
        correction = "" if attempt == 0 else (
            f"\n\n第 {attempt} 次自动修复：上一次响应不是完整合法的 JSON。"
            "保留任务约束，省略不必要解释，严格生成一份完整且闭合的 JSON；不要复述失败输出。"
        )
        transport_retries = _repair_attempts("transport_retries", 1)
        for transport_attempt in range(transport_retries + 1):
            try:
                response = llm.invoke(
                    [("system", system_prompt + correction), ("user", user_prompt)],
                    response_format={"type": "json_object"},
                )
                break
            except Exception as exc:
                if (
                    transport_attempt >= transport_retries
                    or not _is_transient_transport_error(exc)
                ):
                    raise
        try:
            return extract_json_object(_content(response))
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
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
    payload = _invoke_json(
        BRIEF_SYSTEM_PROMPT,
        "【用户原始输入】\n"
        f"{user_prompt}\n\n"
        "【用户单独填写的初始场景覆盖项】\n"
        f"{scene_override or '未填写，由系统从题材中生成'}",
        temperature=0.2,
        max_tokens=4000,
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
        f"{WORLD_SYSTEM_PROMPT}\n\n{WORLD_JSON_INSTRUCTION}",
        "【用户原始输入——最高约束】\n"
        f"{user_prompt}\n\n"
        "【结构化约束账本】\n"
        f"{json.dumps(brief.to_dict(), ensure_ascii=False, indent=2)}",
        temperature=0.7,
        max_tokens=10000,
    )
    world = WorldSpec.from_mapping(payload)
    if brief.requested_opening_scene:
        world.opening_scene = brief.requested_opening_scene
    return world


def generate_character_specs(
    user_prompt: str,
    brief: ScenarioBrief,
    world: WorldSpec,
) -> list[CharacterSpec]:
    payload = _invoke_json(
        f"{CHARACTER_SYSTEM_PROMPT}\n\n{CHARACTER_JSON_INSTRUCTION}",
        "【用户原始输入——所有明确细节均为最高级约束】\n"
        f"{user_prompt}\n\n"
        "【结构化约束账本】\n"
        f"{json.dumps(brief.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        "【结构化世界设定——导演信息只用于分配正确的私有档案】\n"
        f"{json.dumps(asdict(world), ensure_ascii=False, indent=2)}",
        temperature=0.75,
        max_tokens=16000,
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
    repair_prompt = """你是 SceneChat 结构化设定修复器。不得修改用户约束账本。只修复校验错误，并输出完整 JSON：
{
  "world": {与输入 WorldSpec 完全相同的字段},
  "characters": [{与输入 CharacterSpec 完全相同的字段}],
  "warnings": ["必要的非阻断说明"]
}
所有 covered_constraint_ids 必须真实对应其落实位置。公共世界仍不得包含导演秘密。
如果错误涉及规则运行时，必须补全 phase_specs、rules、state_schema 和 termination_rules。termination_rules.kind 只能从 faction_eliminated、faction_parity、world_equals、all_goals_completed、all_active_at_location、all_of、any_of、manual 中选择，禁止创造 ai_win、human_win、team_win 等新 kind；胜方只能写入 winner。每个非 event_only 阶段必须在 allowed_action_types 中保留 pass、observe、speak、act 至少一种安全兜底。advance_when=manual 且存在 next_phase 时，必须提供一个作用于该阶段、effect 为 set_phase 到 next_phase 的可执行规则。阵营胜负场景必须用带 winner 的结构化结束规则覆盖胜负结果。move、vote、inspect、protect、eliminate、poison、heal 的标准效果以及能力次数扣减由 Resolver 内置执行，对应 rule/ability 的 effects 留空；只有额外的题材状态变化才能使用与行动类型匹配的 set_world、increment_world、set_resource、consume_resource、set_goal_status、set_relationship、set_phase 等 effect。不要退回自然语言规则代替结构化字段。"""
    payload = _invoke_json(
        repair_prompt,
        "【用户原始输入】\n"
        f"{user_prompt}\n\n"
        "【不可修改的约束账本】\n"
        f"{json.dumps(package.brief.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        "【校验错误】\n"
        f"{json.dumps(issues, ensure_ascii=False, indent=2)}\n\n"
        "【待修复结果】\n"
        f"{json.dumps(package.to_dict(), ensure_ascii=False, indent=2)}",
        temperature=0.2,
        max_tokens=16000,
    )
    world_payload = payload.get("world") if isinstance(payload.get("world"), dict) else {}
    character_payload = payload.get("characters")
    characters = [
        CharacterSpec.from_mapping(item, index)
        for index, item in enumerate(character_payload or [], start=1)
        if isinstance(item, dict)
    ]
    return ScenarioPackage(
        brief=package.brief,
        world=WorldSpec.from_mapping(world_payload),
        characters=characters,
        warnings=[str(item) for item in payload.get("warnings") or []],
    )


def generate_scenario_package(user_prompt: str, scene_override: str = "") -> ScenarioPackage:
    brief = generate_scenario_brief(user_prompt, scene_override)
    world = generate_world_spec(user_prompt, brief)
    characters = generate_character_specs(user_prompt, brief, world)
    package = ScenarioPackage(brief=brief, world=world, characters=characters)
    normalize_scenario_phase_references(package)
    issues = validate_scenario_package(package)
    for _ in range(scenario_semantic_repair_attempts()):
        if not issues:
            break
        package = repair_scenario_package(user_prompt, package, issues)
        normalize_scenario_phase_references(package)
        issues = validate_scenario_package(package)
    if issues:
        raise ScenarioValidationError(issues)
    return package
