from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from Character import CHARACTER_DESIGN_PROMPT
from .config import config_int
from .build_control import BuildControl, CURRENT_BUILD
from .scenario_patch import apply_scenario_patch
from .errors import SceneChatError
from .recovery import fingerprint, is_transient, retry_delay, repair_local_fields
from .prompt_library import prompt_contract

from .providers import get_generation_chat_model
from .scenario import (
    CharacterSpec,
    ScenarioBrief,
    ScenarioPackage,
    ScenarioValidationError,
    StateFieldSpec,
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
    {"name":"自由推进", "scheduler":"round_robin", "allowed_action_types":["speak","pass"], "actor_roles":[], "advance_when":"manual", "next_phase":"", "event_only":false, "opening_min_cycles":1}
  ],
  "rules": [
    {"id":"rule-1", "description":"普通发言", "action_type":"speak", "phases":["自由推进"], "allowed_roles":[], "target_scope":"none", "priority":0, "effect_mode":"rule", "behavior_template":"", "effects":[], "visibility":["public"]}
  ],
  "termination_conditions": ["自然结束或胜负条件"],
  "termination_rules": [
    {"id":"end-1", "kind":"faction_eliminated|faction_parity|world_equals|entity_equals|all_goals_completed|all_active_at_location|all_of|any_of|manual", "description":"结束说明", "phases":["允许判定胜负的结算/公布阶段；自由场景可为空"], "faction":"可选阵营", "opposing_factions":[], "target":"entity_equals 的实体 ID", "key":"相等比较使用的状态键", "value":"目标值，必须保持实际类型", "location":"all_active_at_location 使用的地点", "winner":"胜方", "conditions":[]}
  ],
  "fixed_canon": ["从用户约束账本继承的必须发生内容"],
  "target_beats": ["需要跟踪但不强迫具体角色选择的节点"],
  "beat_specs": [
    {"id":"beat-1","description":"可观察、可判定是否已经发生的剧情节点","required":false,"weight":1,"prerequisites":[],"phase_hint":"可选阶段名","completion_mode":"narrative","completion_conditions":[],"resolution_signals":["已提交角色行动中的具体证据"]}
  ],
  "covered_constraint_ids": ["本世界设定已落实的约束 ID"]
}

多个条件必须同时成立时，使用一个 kind=all_of 的顶层规则，把原子规则放入 conditions；任选其一时使用 any_of。不得把“且/全部/同时”条件拆成多个并列顶层规则，因为顶层规则之间按任一满足处理。

新场景使用 execution_version=2：行动名称不隐含效果，必须显式填写 effects。rule.effect_mode=rule（默认，仅规则结果）、ability（仅能力结果）、stack（明确叠加）。能力使用次数与 consume_resource 成本始终校验扣除，不要重复声明 consume_ability。规则重叠用不同 priority 选最大者，禁止依赖顺序。没有规则时使用能力自身效果。会议 vote 只记录提案投票，不淘汰角色；医治 heal 不等于复活；inspect 只返回规则准许的事实。只有用户设定确实要求传统桌游效果时，才可显式选 behavior_template=legacy_tabletop，此时 effects 留空；不得根据题材名称强制套模板。
target_scope 除角色 same_location/any_active/character/self/none，还支持 location/object/proposal。物品或提案声明在 entities 对象中，例如 {"proposal-A":{"kind":"proposal","state":{"passed":false}}}；通过 set_entity(target,key,value) 修改已声明的同类型字段。地点用 locations。资源在角色 resources 声明。move_agent 的 target=$actor、value=$target；record_vote 的 target=$target。不能发明运算符或运行代码；复杂统计若无法由已有操作精确表达，不能伪称支持。
观众策略 audience_policy=limited|omniscient、reveal_policy=preserve_suspense|allow_reveal 应遵从用户要求，默认 limited + preserve_suspense。全知读者镜头仍是 audience_only，不能因此让角色知晓秘密。

allowed_action_types 只能列出本阶段确实允许的行动；允许弃权或观察时才加入 pass/observe。强制投票阶段可以只允许 vote，不能为了通过校验而虚构 pass：pass 不记录选票，也不能满足 all_active_voted。生成失败后由运行时停止无效行动，不靠添加虚假动作掩盖机制缺口。
advance_when=manual 表示阶段可无限持续；如果同时填写 next_phase，必须提供一条在本阶段可执行且包含 set_phase 到 next_phase 的规则，不能只在自然语言中说“之后进入下一阶段”。阶段切换必须是独立、明确的 act 行动；不能把它设为普通 speak 的高优先级规则，否则任何发言都会立刻跳阶段。初次见面的多人场景，不能在只听到一人发言后就切到表决，应让人物先有基本认识、试探和不同关注点。阵营对抗或比赛场景的 termination_rules 必须明确 winner，并能区分主要胜负结果。
有夜间技能、治疗、反制或结算顺序时，termination rule 必须用 phases 限制到结算/公布阶段，不能在中间行动后提前判胜。

状态 effect 仅允许 set_world、increment_world、set_entity、move_agent、set_agent_status、set_resource、consume_resource、set_goal_status、set_relationship、record_vote、clear_votes、set_phase、add_known_fact、protect_agent、clear_protections。模板中的 $actor、$target、$value 会在运行时由 Resolver 安全替换。
仅这三个占位符可用；$max_votes_target、$winner 等不会计算票数，也不会被替换。若用户确实要求狼人杀式“全员投票后最高票角色出局，平票无人出局”，可在 vote 角色规则明确使用 behavior_template=legacy_tabletop 且 effects=[]；不要另造 host 角色或在 event_only 结算阶段放一条需要角色执行的规则。环境事件只能播报已经由投票规则提交的结果。普通提案表决仍使用下方声明式 settle_votes，不能误套淘汰模板。
faction_eliminated 结束规则只有角色实际变成不可行动时才能触发。record_vote 只是记录选票，settle_votes 只修改提案布尔状态，二者都不会淘汰角色；若故事必须按投票淘汰角色，就要有明确可执行的淘汰效果（例如符合用户设定的 legacy_tabletop），不可用一条提案表决规则冒充角色淘汰结算。同一阶段同名行动如同时面向人物与提案，应明确区分 target_scope，并确保目标与效果一致。
每条结构化结束规则都要有真实、可执行的状态来源：如果 entity_equals 指向某个实体状态，必须有角色规则、能力或合法结算效果会更新同一实体与状态键；不能只把初始 false/true 当成“胜负统计”，进入公布阶段就预定胜者。隐藏身份不能靠只为真实卧底建立公开实体 ID 来暗示答案；若结局取决于隐藏阵营，优先用实际淘汰/存活状态的通用条件，而不是在公共世界中硬编码卧底姓名。
提案投票还可使用 settle_votes：target=提案 ID，key=该提案的布尔状态字段，amount=所需同意票数。在全部可行动角色投票后按该提案得票数写 true/false；须放在 record_vote 后、clear_votes 或 set_phase 前。没有全员投票时不结算。不支持表达式、条件脚本或任意统计。世界环境变量若允许旁白更新，state_schema.mutable_by 必须包含 director；关键目标结果只允许 resolver，避免旁白越权宣布结果。
实体结果作为结束条件时使用 termination_rules.kind=entity_equals，并填写 target=实体 ID、key=状态键、value=准确类型的目标值；可以放进 all_of/any_of。

beat_specs 必须与 target_beats 一一对应并使用稳定 ID。用户明确要求必须发生的节点标 required=true；普通期望节点不得标成硬约束。description 和 resolution_signals 必须描述可观察结果，不能要求某角色违背自主判断作出指定选择；prerequisites 只引用前面已声明的 beat ID。
每个节点声明 completion_mode=state|narrative。可执行结果必须用 state 和 completion_conditions（所有条件同时满足）：world_equals(key,value)、entity_equals(target,key,value)、agent_location(target,value)、goal_equals(target,key,value)，每个条件使用 kind 字段。状态键必须声明，目标必须存在。找到线索不等于找到失踪者，治疗步骤不等于治愈。承诺、和解等使用 narrative，写清 resolution_signals，之后由已提交角色事件证据核验；旁白声称完成不是证据。开放场景无需虚构固定终点或节点。
all_eligible_acted、all_active_voted、phase_advanced 是阶段推进方式或观察信号，不是 beat 的 completion_conditions.kind；若无上述四种可声明状态结果，节点须使用 narrative 并列出实际角色行动的核验信号。

特别注意：公共世界会直接成为所有角色的知识。任何并非所有角色都知道的信息只能进入 director_notes_markdown 或带精确非 public scope 的 facts。桌游、审判、比赛等规则题材必须给出可执行 phase_specs、rules 与 termination_rules；自由谈话可以保持精简。天亮公布、结算、广播等没有角色行动的阶段必须设置 scheduler=event_first、advance_when=after_event、event_only=true，并给出 next_phase；唯一例外是有适用结构化结束规则的终局阶段，可以不设 next_phase。

relationship_dimensions 应选择 2–5 个对本场景真正有用、含义互不重复的主观关系维度，必须兼容题材而不是默认套用猜疑游戏：战斗可使用敌对/协作、威胁判断、敬重或服从；职场可使用合作、可靠判断、影响力；家庭或情感场景可使用亲密、信赖、依赖或边界感。不要把战斗胜负、生命值、距离等客观状态伪装成关系维度。若题材没有特殊需要，可使用 cooperation、confidence、regard 三个通用维度。"""

RUNTIME_GENERATION_GUIDANCE = """
set_phase 使用 value 存放目标阶段原文，例如 {"op":"set_phase","value":"秘密投票"}；target 不是目标阶段字段。不得写 value=null 再把阶段填入 target。
结构化协议必须逐字段遵循：effects 是数组，每项操作的 op/key/target/value/amount 均在同一层，禁止嵌套 params、arguments、path 或自造操作（例如 check_condition_and_set）。set_agent_status 只允许 key=alive/active 和布尔 value；其他人物数值放 resources。实体结构必须是 {"实体ID":{"kind":"object或proposal","state":{"状态键":初始值}}}，不得把 state 改叫 properties。地点在 locations、opening_scene 与人物 initial_location 中使用一致的原文名称，不要混用英文 ID 和中文名。
V2 提案表决的最小操作示例（仅在用户要求此机制时使用，不能强加给其他场景）：先声明 entities={"proposal-A":{"kind":"proposal","state":{"passed":false}}}，对应 rule.action_type=vote、target_scope=proposal、effect_mode=rule、effects=[{"op":"record_vote","target":"$target"},{"op":"settle_votes","target":"$target","key":"passed","amount":2}]。两人投票通过的结束规则为 {"id":"end","kind":"entity_equals","target":"proposal-A","key":"passed","value":true,"description":"提案通过"}。不得增加未经支持的条件脚本或把 passed 当成角色状态。
运行可达性与人物命名约束：
- actor_roles/allowed_roles 为空数组表示所有可行动角色；非空时必须与实际 CharacterSpec.role 一致，禁止虚构 all_active 职业或名单外的 host。规则 action_type 必须在其阶段 allowed_action_types 中。
- 主持流程由公开环境事件或已有角色完成，不能为了主持额外增加用户未要求的角色。有限回合的讨论优先 all_eligible_acted；只有确实允许无限讨论时才用 manual。manual 的退出规则必须有真实有权执行的角色和被允许的动作。
- 短输入优先设计可复用的一轮阶段循环，不预先复制 Round 1/2/3 三套同构阶段与规则；轮数由运行时推进。若确需手动阶段，先确认退出动作和 set_phase 规则都在该角色阶段可执行，不能把角色规则放到 event_only 阶段。
- 普通环境事件阶段使用 after_event 并明确 next_phase；事件阶段链必须能回到角色行动阶段。终局可以没有 next_phase，但必须有在该阶段适用的结构化结束规则；不得依赖旁白永远循环。
- opening_min_cycles 仅用于 all_eligible_acted 阶段第一次进入时的最低角色行动周期数（1-3），回到同一阶段后仍按一轮完成；短设定里陌生人初遇就要表决时通常给首轮两次互动机会，避免七个人各说一句就立刻投票。用户明确规定流程时照其规则，不要覆盖。
- 使用 legacy_tabletop 的角色投票已经在最后一张合法选票时由引擎结算淘汰。随后如需“公布/结算”场景，应设 event_only=true、scheduler=event_first、advance_when=after_event；不要安排所有角色轮流执行 settle_votes 或“等待结果”，也不要制造与真实淘汰无关的 proposal.passed 占位状态来证明已结算。
- world_equals 是类型严格的相等比较，不执行 >=、<= 或表达式；存活阵营数量的胜负使用 faction_eliminated/faction_parity，不能依赖无人维护的计数字段。
- 普通问答或质疑使用 speak + question/challenge，inspect 仅用于场景明确授权的事实查验能力，不得让普通玩家提问就读取秘密阵营。
- vote 规则必须记录真实选票（record_vote 或明确符合题材的 legacy_tabletop），不得另设一条 vote 规则仅写 game_status/winner 等胜负状态。胜负计算只放 termination_rules；不能凭某一张选票无条件宣布阵营获胜。规则数量以能走通一轮为准，不用多条同 action_type 的高优先级伪规则装饰流程。
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
      "personality": "稳定倾向、真实牵挂及可能使其动摇的处境；不是固定口癖或职业标签",
      "goals": ["目标、顾虑和可接受代价"],
      "core_beliefs": ["只有当用户设定或人物经历确实支持时，填写会持续影响选择的价值信念；否则为空数组"],
      "knowledge": {
        "确定知道": "...",
        "不知道": "...",
        "怀疑": "...",
        "可能错误相信": "..."
      },
      "decision_logic": "在本题材关键局面中如何随新事实、压力和关系改变选择；不要给固定万能话术",
      "voice_profile": {
        "register": "自然口语、正式、古典、粗粝、克制等；优先使用用户明确设定",
        "sentence_length": "短句为主|中等|长短交替",
        "directness": 0.5,
        "emotional_expressiveness": 0.5,
        "politeness": 0.5,
        "humor_style": "没有时为空字符串",
        "rhetorical_habits": ["用户明确指定或确有必要时才填写；不能是每轮都要说的标志性比喻或句子"],
        "avoidances": ["该角色不会使用的措辞或表达方式"],
        "vocabulary_hints": ["用户明确要求或时代语境确有需要时才填写；普通人物可为空数组"]
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

角色数量必须与约束账本的 requested_character_count 完全一致；各阵营人数也必须逐一满足 locked 约束，不能只对上总人数。例如七人中两名 AI，faction=AI 必须恰好出现两次，其余五人不是 AI，private_identity 与彼此已知秘密也要同步。公开字段不能泄露 private_identity、faction、私有知识、隐藏阵营或秘密任务。能力必须结构化且引用真实阶段；普通交谈能力可以为空。夜袭、查验、守护等秘密技能默认 director_only；只有在场角色可直接观察的技能才标 public。execution_version=2 中行动名称不隐含效果，能力必须显式声明 effects，并与规则的 effect_mode 配合；只有规则明确采用 behavior_template=legacy_tabletop 时才使用相应内置效果。

personality 与 decision_logic 是影响行动的两个独立字段：前者写此人面对风险、冲突和未知时的稳定倾向及来源，后者写这些倾向与目标在本场景里如何改变具体选择。不能只重复职业、阵营或 voice_profile；同一事项交给不同人物，可以有不同的接受条件、拒绝边界、冒险程度或信息披露策略，也允许偶尔犯与惯常形象不一致但有情境原因的错。
短输入的补全也应保留信息不对称与动机不对称：不要把所有人物写成“表面指责、暗中也心虚”的镜像，或者给每人安排一件恰好对称的隐情。private_identity 可以是“无额外秘密”。decision_logic 写可随事实变化的判断边界，不要预定“被揭穿后最终承认/妥协”的剧情。未被世界或人物档案确立的具体旧事、设备状况或责任归属，应留待后续可见事件核验。
如果用户设定是陌生人初遇，不得补共同回忆、此前结盟或相互熟悉的判断。高压场景也不要把每个人的 personality、goals、decision_logic 都写成单一胜负任务的分支算法：各人可能先关心规则可信度、身体安全、如何与陌生人共处、家人或自身处境；只挑与此人当下有关的部分，不用随机家常话装点。少数有具体理由的人可以从第一轮就质疑、施压或挑衅，但不让全员开场就像完成过多轮分析。性格写可以改变的倾向，不写“第三轮必质问/后期必一击命中”的预定剧本。
voice_profile 只是倾向，不是台词模板或必须兑现的差异配额。用户明确写过语气、时代用语、措辞禁忌或表达习惯时必须原样落实；用户未写时以自然口语为基线，不为每个人硬造风格。职业术语、文化标记、隐喻与幽默只有在此人此刻真的会用时才出现；rhetorical_habits、humor_style、vocabulary_hints 没有可靠依据时留空。差异首先来自具体处境与选择，而不是词汇表。

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


def _issue_anchor(issue: str) -> str:
    """Identify the failing component, not the validator's changing wording.

    A patch may turn an unsupported beat condition into a missing condition;
    that beat is still broken, but other fixed components are real progress.
    Unknown messages stay literal so new, unclassified failures are rejected.
    """
    text = str(issue)
    conflict = re.match(r"规则 ([\w-]+/[\w-]+) 匹配冲突", text)
    if conflict:
        return f"rule-conflict:{conflict.group(1).split('/')[0]}"
    match = re.match(r"(?:规则/能力|规则|节点|结束规则)\s*[“\"]?([\w-]+)", text)
    if match:
        kind = "beat" if text.startswith("节点") else "rule"
        return f"{kind}:{match.group(1)}"
    match = re.match(r"(?:手动阶段|规则型阶段|投票阶段|角色阶段|环境事件阶段|阶段)[“\"]([^”\"]+)[”\"]", text)
    if match:
        return f"phase:{match.group(1)}"
    return f"literal:{text}"


def _semantic_progress(before: list[str], after: list[str], changed: bool) -> bool:
    if not changed or len(after) >= len(before):
        return False
    # Do not trade one failing component for a new one just to lower a count.
    # In particular, a lost rule or a newly conflicting rule is a regression.
    return {_issue_anchor(item) for item in after} <= {_issue_anchor(item) for item in before}


def _repair_mechanical_world_fields(world: WorldSpec) -> tuple[WorldSpec, list[str]]:
    """Fix unambiguous schema omissions and byte-identical duplicated rules."""
    repaired = deepcopy(world)
    changes = []
    seen = set()
    unique_rules = []
    for rule in repaired.rules:
        signature = fingerprint(asdict(rule))
        if signature in seen:
            changes.append(f"重复规则 {rule.id}")
            continue
        seen.add(signature)
        unique_rules.append(rule)
    repaired.rules = unique_rules
    missing = {}
    for rule in repaired.rules:
        for effect in rule.effects:
            if effect.op not in {"set_world", "increment_world"} or not effect.key or effect.key in repaired.state_schema:
                continue
            initial = repaired.initial_state.get(effect.key)
            if type(initial) is bool:
                value_type = "boolean"
            elif type(initial) is int or effect.op == "increment_world":
                value_type = "integer"
            elif initial is None and type(effect.value) is bool:
                value_type = "boolean"
            elif initial is None and type(effect.value) is int:
                value_type = "integer"
            elif initial is None and (isinstance(effect.value, str) and effect.value != "$value"):
                value_type = "string"
            elif type(initial) is str:
                value_type = "string"
            else:
                continue
            previous = missing.get(effect.key)
            if previous and previous[0] != value_type:
                missing[effect.key] = None
            elif effect.key not in missing:
                missing[effect.key] = (value_type, list(rule.visibility))
    defaults = {"boolean": False, "integer": 0, "string": ""}
    for key, specification in missing.items():
        if specification is None:
            continue
        value_type, visibility = specification
        repaired.state_schema[key] = StateFieldSpec(
            key, value_type=value_type,
            visibility=visibility or ["public"], mutable_by=["resolver"],
        )
        repaired.initial_state.setdefault(key, defaults[value_type])
        changes.append(f"状态字段 {key}")
    return repaired, changes


def _repair_generic_relationship_targets(package: ScenarioPackage, user_prompt: str) -> list[str]:
    """A group label is a trait, not a reference to a nonexistent character."""
    names = {character.name for character in package.characters}
    generic = {"其他玩家", "其他人", "所有玩家", "大家", "众人", "全体"}
    changes = []
    for character in package.characters:
        for target in set(character.relationships) | set(character.relationship_facets):
            if target in names or target not in generic or target in user_prompt:
                continue
            attitude = character.relationships.pop(target, "")
            character.relationship_facets.pop(target, None)
            if attitude and attitude not in character.personality:
                character.personality = (character.personality + f" 对{target}的初始态度：{attitude}").strip()
            changes.append(f"{character.name}:{target}")
    return changes


def _repair_world_phase_beats(world: WorldSpec) -> tuple[WorldSpec, list[str]]:
    """Reinterpret phase observations as evidence-backed narrative beats.

    A phase counter is not a world/entity state field. This narrow correction
    preserves the requested milestone and never invents a state mutation.
    Mixed or unknown conditions remain for model-assisted repair.
    """
    phase_conditions = {"all_eligible_acted", "all_active_voted", "phase_advanced"}
    repaired = deepcopy(world)
    corrected = []
    legacy_character_vote = any(
        rule.action_type == "vote" and rule.behavior_template == "legacy_tabletop"
        and rule.target_scope in {"character", "any_active", "same_location"}
        for rule in repaired.rules
    )
    proposal_vote = any(
        rule.action_type == "vote" and rule.target_scope == "proposal"
        for rule in repaired.rules
    )
    for beat in repaired.beat_specs:
        conditions = beat.completion_conditions
        phase_observation = bool(conditions) and all(
            isinstance(item, dict) and item.get("kind") in phase_conditions
            for item in conditions
        )
        pseudo_vote_proposal = (
            legacy_character_vote and not proposal_vote and bool(conditions)
            and ("投票" in beat.description or "淘汰" in beat.description)
            and all(isinstance(item, dict) and item.get("kind") == "entity_equals"
                    and isinstance(repaired.entities.get(str(item.get("target"))), dict)
                    and repaired.entities[str(item.get("target"))].get("kind") == "proposal"
                    for item in conditions)
        )
        if beat.completion_mode == "state" and (phase_observation or pseudo_vote_proposal):
            beat.completion_mode = "narrative"
            beat.completion_conditions = []
            if not beat.resolution_signals and beat.description:
                beat.resolution_signals = [beat.description]
            corrected.append(beat.id)
    return repaired, corrected


def _prune_unreachable_noop_rules(world: WorldSpec) -> tuple[WorldSpec, list[str]]:
    """Discard inert pseudo-event actions, never a rule with executable effects.

    Event-only phases are narrated by the director; they cannot schedule a
    character action. A rule bound only there with neither effects nor a
    behavior template cannot perform the advertised settlement.
    """
    repaired = deepcopy(world)
    phases = {phase.name: phase for phase in repaired.phase_specs}
    kept = []
    removed = []
    for rule in repaired.rules:
        selected = [phases[name] for name in rule.phases if name in phases]
        executable_elsewhere = any(
            not phase.event_only and rule.action_type in phase.allowed_action_types
            for phase in phases.values()
        )
        inert_event_rule = (
            rule.phases and len(selected) == len(rule.phases)
            and all(phase.event_only for phase in selected)
            and not executable_elsewhere and not rule.effects
            and not rule.behavior_template
        )
        if inert_event_rule and len(repaired.rules) > 1:
            removed.append(rule.id)
        else:
            kept.append(rule)
    if kept:
        repaired.rules = kept
    return repaired, removed


def _repair_ambiguous_phase_transitions(world: WorldSpec) -> tuple[WorldSpec, list[str]]:
    """Separate a deliberate phase change from ordinary speech.

    A higher-priority ``speak`` rule with ``set_phase`` otherwise fires on
    *every* conversation turn. Announcement-only phases belong to the
    director, not to an arbitrary actor selected by the scheduler.
    """
    if world.execution_version != 2:
        return world, []
    repaired = deepcopy(world)
    changed = []
    event_markers = ("announcement", "resolution", "公布", "结算", "广播", "揭晓")
    removed_rule_ids = set()
    legacy_vote_phases = {
        name for rule in repaired.rules
        if rule.action_type == "vote" and rule.behavior_template == "legacy_tabletop"
        and rule.target_scope in {"character", "any_active", "same_location"}
        for name in rule.phases
    }
    referenced_entities = {
        str(condition.get("target"))
        for beat in repaired.beat_specs for condition in beat.completion_conditions
        if isinstance(condition, dict) and condition.get("target")
    }
    def termination_targets(rule):
        targets = {rule.target} if rule.target else set()
        for child in rule.conditions:
            targets.update(termination_targets(child))
        return targets
    for terminal in repaired.termination_rules:
        referenced_entities.update(termination_targets(terminal))
    for phase in repaired.phase_specs:
        phase_rules = [rule for rule in repaired.rules if phase.name in rule.phases]
        previous_legacy_vote = any(
            prior.name in legacy_vote_phases and prior.next_phase == phase.name
            for prior in repaired.phase_specs
        )
        unused_vote_settlement = (
            previous_legacy_vote and bool(phase_rules)
            and all(rule.phases == [phase.name] and rule.action_type == "settle_votes"
                    and rule.effects and all(
                        effect.op == "set_entity"
                        and effect.target in repaired.entities
                        and isinstance(repaired.entities[effect.target], dict)
                        and repaired.entities[effect.target].get("kind") == "proposal"
                        and effect.target not in referenced_entities
                        and not any(effect.target == other_effect.target
                                    for other_rule in repaired.rules if other_rule not in phase_rules
                                    for other_effect in other_rule.effects)
                        for effect in rule.effects
                    ) for rule in phase_rules)
        )
        if (phase.scheduler == "event_first" and phase.next_phase
                and any(marker in phase.name.lower() for marker in event_markers)
                and phase_rules and (unused_vote_settlement or all(
                    rule.phases == [phase.name] and rule.effects
                    and all(effect.op == "set_phase" and effect.value == phase.next_phase
                            for effect in rule.effects)
                    for rule in phase_rules
                ))):
            phase.event_only = True
            phase.advance_when = "after_event"
            phase.allowed_action_types = []
            removed_rule_ids.update(rule.id for rule in phase_rules)
            changed.append(phase.name)
            continue
        if phase.event_only or phase.advance_when != "manual" or not phase.next_phase:
            continue
        ordinary_speech = any(
            rule.action_type == "speak" and not rule.effects and not rule.behavior_template
            and (not rule.phases or phase.name in rule.phases)
            for rule in repaired.rules
        )
        if not ordinary_speech:
            continue
        for rule in phase_rules:
            if (rule.action_type == "speak" and rule.target_scope == "none"
                    and rule.effects and all(
                        effect.op == "set_phase" and effect.value == phase.next_phase
                        for effect in rule.effects
                    )):
                rule.action_type = "act"
                if "act" not in phase.allowed_action_types:
                    phase.allowed_action_types.append("act")
                changed.append(rule.id)
    if removed_rule_ids:
        repaired.rules = [rule for rule in repaired.rules if rule.id not in removed_rule_ids]
    return repaired, changed


def _repair_vote_contract(world: WorldSpec, user_prompt: str) -> tuple[WorldSpec, list[str]]:
    """Remove model-invented ballot shortcuts without changing explicit rules."""
    if world.execution_version != 2:
        return world, []
    repaired = deepcopy(world)
    changed = []
    formal_votes = {
        phase.name for phase in repaired.phase_specs
        if phase.advance_when == "all_active_voted"
    }
    explicit_abstention = bool(re.search(r"弃权|不投票|abstain", user_prompt, re.I))
    explicit_prevote = bool(re.search(r"预投票|意向票|提名票|nomination vote|straw poll", user_prompt, re.I))
    kept_rules = []
    for rule in repaired.rules:
        if rule.action_type != "vote" and rule.effects and all(
            effect.op == "record_vote" for effect in rule.effects
        ):
            rule.effects = []
            changed.append(rule.id)
        premature_ballot = (
            not explicit_prevote and rule.action_type == "vote"
            and rule.effects and all(effect.op == "record_vote" for effect in rule.effects)
            and any(phase.name in rule.phases and phase.next_phase in formal_votes
                    for phase in repaired.phase_specs)
            and any(other.action_type == "vote"
                    and any(name in formal_votes for name in other.phases)
                    for other in repaired.rules)
        )
        if premature_ballot:
            changed.append(rule.id)
            continue
        if not explicit_abstention and rule.action_type in {"pass", "observe"}:
            removed = [name for name in rule.phases if name in formal_votes]
            if removed:
                rule.phases = [name for name in rule.phases if name not in formal_votes]
                changed.append(rule.id)
                if not rule.phases:
                    continue
        kept_rules.append(rule)
    repaired.rules = kept_rules
    if not explicit_abstention:
        for phase in repaired.phase_specs:
            if phase.name in formal_votes:
                before = list(phase.allowed_action_types)
                phase.allowed_action_types = [
                    action for action in before if action not in {"pass", "observe"}
                ]
                if phase.allowed_action_types != before:
                    changed.append(phase.name)
    if not explicit_prevote:
        for phase in repaired.phase_specs:
            if phase.next_phase in formal_votes and "vote" in phase.allowed_action_types:
                phase.allowed_action_types.remove("vote")
                changed.append(phase.name)
    return repaired, changed


def _prune_shadow_vote_wins(world: WorldSpec) -> tuple[WorldSpec, list[str]]:
    """Drop redundant vote-shaped win declarations, never real ballot rules."""
    if world.execution_version != 2 or not world.termination_rules:
        return world, []
    from .mechanics import records_vote
    repaired = deepcopy(world)
    valid_votes = [rule for rule in repaired.rules if rule.action_type == "vote" and records_vote(rule)]
    removed = []
    kept = []
    for rule in repaired.rules:
        phase_overlap = any(
            not rule.phases or not valid.phases or bool(set(rule.phases) & set(valid.phases))
            for valid in valid_votes
        )
        pseudo_win = (
            rule.action_type == "vote" and not records_vote(rule) and phase_overlap
            and bool(rule.effects)
            and all(effect.op == "set_world"
                    and any(marker in str(effect.key).lower()
                            for marker in ("status", "winner", "victory", "胜负", "获胜"))
                    for effect in rule.effects)
        )
        if pseudo_win:
            removed.append(rule.id)
        else:
            kept.append(rule)
    repaired.rules = kept
    return repaired, removed


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
        repeated = max(failures["outputs"].values(), default=0) >= 2
        exhausted = failures["count"] >= 6
        if repeated or exhausted:
            issue = str(failures.get("last_issue") or "").strip()
            raise SceneChatError("structured_output_stalled",
                                 ("模型重复返回相同的无效结构，已停止自动重试。" if repeated else
                                  "累计六次结构化响应仍未通过校验，已停止自动重试。")
                                 + (f"最近的具体问题：{issue}。" if issue else "")
                                 + "请检查模型输出与规则协议；检查点已保留。",
                                 stage=control.stage, status_code=422)
    check_stalled()
    repairs = _repair_attempts("json_repair_retries", 2) if allow_json_repair else 0
    last_error: Exception | None = None
    request_limit = config_int("scenario", "max_requests_per_step", 4, minimum=1, maximum=8)
    for attempt in range(repairs + 1):
        control.check()
        correction = "" if attempt == 0 else (
            f"\n\n第 {attempt} 次自动修复：上一次响应未通过 JSON、结构或可执行世界规则校验。"
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
            failures["last_issue"] = f"{type(exc).__name__}: {str(exc)[:240]}"
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


def _ensure_short_input_opening_room(world: WorldSpec, brief: ScenarioBrief, user_prompt: str) -> bool:
    """Allow first-contact scenes to breathe without lengthening later rounds."""
    if (brief.input_mode != "short" or (brief.requested_character_count or 0) < 3
            or re.search(r"(?:只|仅|每人).{0,5}(?:一轮|一次|一句).{0,5}(?:发言|讨论)|立即投票|马上投票", user_prompt)):
        return False
    if not world.phase_specs:
        return False
    opening = world.phase_specs[0]
    if (opening.event_only or opening.advance_when != "all_eligible_acted"
            or "speak" not in opening.allowed_action_types or not opening.next_phase):
        return False
    vote_phase = next((phase for phase in world.phase_specs if phase.name == opening.next_phase), None)
    if vote_phase is None or "vote" not in vote_phase.allowed_action_types:
        return False
    if opening.opening_min_cycles >= 2:
        return False
    opening.opening_min_cycles = 2
    return True


def generate_world_spec(user_prompt: str, brief: ScenarioBrief) -> WorldSpec:
    from .mechanics import validate_world_structure
    control = CURRENT_BUILD.get()
    if control is None:
        local_control = BuildControl()
        local_control.begin_step("world")
        with local_control.activate():
            return generate_world_spec(user_prompt, brief)
    ledger = control.recovery.setdefault("world_semantic", {
        "attempts": 0, "no_progress": 0, "history": [],
    })
    if isinstance(ledger.get("candidate"), dict):
        world = WorldSpec.from_mapping(ledger["candidate"])
        control.progress(reason="已恢复上一份世界候选，继续修复而非重新生成")
    else:
        payload = _invoke_json(
            f"{prompt_contract('scenario')}\n\n{WORLD_JSON_INSTRUCTION}\n\n{RUNTIME_GENERATION_GUIDANCE}",
            "【用户原始输入——最高约束】\n"
            f"{user_prompt}\n\n"
            "【结构化约束账本】\n"
            f"{json.dumps(brief.to_dict(), ensure_ascii=False, indent=2)}",
            temperature=0.45,
            max_tokens=10000,
            validate_payload=lambda candidate: WorldSpec.from_mapping({**candidate, "execution_version": 2}),
        )
        payload["execution_version"] = 2
        world = WorldSpec.from_mapping(payload)
    if brief.requested_opening_scene:
        world.opening_scene = brief.requested_opening_scene
    world, repaired_votes = _repair_vote_contract(world, user_prompt)
    world, corrected_beats = _repair_world_phase_beats(world)
    world, removed_rules = _prune_unreachable_noop_rules(world)
    world, removed_vote_wins = _prune_shadow_vote_wins(world)
    world, repaired_transitions = _repair_ambiguous_phase_transitions(world)
    world, mechanical_fixes = _repair_mechanical_world_fields(world)
    opening_expanded = _ensure_short_input_opening_room(world, brief, user_prompt)
    if corrected_beats:
        control.progress(reason="已将阶段推进观察改为需要角色事件证据的叙事节点",
                         local_repaired_beats=corrected_beats)
    if repaired_votes:
        control.progress(reason="已移除非投票动作的伪选票及不能完成结算的默认弃权",
                         local_repaired_vote_contract=repaired_votes)
    if removed_rules:
        control.progress(reason="已删除只挂在环境事件阶段、无任何效果的不可执行规则",
                         local_removed_rules=removed_rules)
    if removed_vote_wins:
        control.progress(reason="胜负由结束规则判定，已移除不会记录选票的伪投票规则",
                         local_removed_vote_wins=removed_vote_wins)
    if repaired_transitions:
        control.progress(reason="已将普通发言和显式阶段切换分离，并把纯公告阶段交由导演处理",
                         local_repaired_transitions=repaired_transitions)
    if opening_expanded:
        control.progress(reason="已为短设定的初次见面增加一次互动周期，后续轮次不延长")
    if mechanical_fixes:
        control.progress(reason="已补齐明确类型的状态声明并移除完全重复的规则",
                         local_mechanical_fixes=mechanical_fixes)
    ledger["candidate"] = asdict(world)
    control.save_recovery()
    issues = validate_world_structure(world)
    total_limit = config_int("scenario", "semantic_total_attempts", 6, minimum=1, maximum=12)

    def check_stalled():
        if ledger["no_progress"] >= 2 or ledger["attempts"] >= total_limit:
            summary = "；".join(issues[:3])[:360]
            raise SceneChatError(
                "scenario_repair_stalled",
                "世界规则局部修复未继续减少错误或已达累计上限；较好的候选已保留。"
                + (f"尚未解决：{summary}。" if summary else "")
                + "请调整设定或模型后重新生成，重复继续不会再次调用修复模型。",
                stage="world", status_code=422,
            )

    for _ in range(scenario_semantic_repair_attempts()):
        if not issues:
            break
        check_stalled()
        control.progress(repair_attempt=ledger["attempts"] + 1, issue_count=len(issues),
                         issues=issues[:30], reason="针对已保存的世界候选修复阶段和规则")
        package = ScenarioPackage(brief=brief, world=world, characters=[])
        patch = _invoke_json(
            "你是 SceneChat 世界规则局部修复器。只输出 JSON 字段补丁："
            '{"changes":[{"path":"/world/phase_specs/0/advance_when","value":"all_eligible_acted"}]}。'
            "path 必须以 /world/ 开头，最多 40 项，禁止修改 execution_version，禁止重写整个 world。"
            "仅修复列出的错误及其直接依赖；不得删除用户明确设定，不能虚构主持角色、脚本或新机制。"
            "manual 且有 next_phase 时必须有本阶段真实可执行的 set_phase 出口；"
            "event_only 阶段只由环境事件推进，不能执行角色规则。若规则仅挂在 event_only 阶段，"
            "先判断它是否与已有可执行规则重复；重复时可替换 /world/rules 数组删除冗余规则，"
            "否则把它修到真正允许该 action_type 的角色行动阶段，并检查效果触发时机；"
            "不得只改阶段名称却让投票结算在第一张票时发生。"
            "有限讨论阶段可用 all_eligible_acted，强制投票阶段可只允许 vote，"
            "不要添加不计票的 pass 冒充结算。",
            "【用户原始输入】\n" + user_prompt
            + "\n【不可修改的约束账本】\n"
            + json.dumps(brief.to_dict(), ensure_ascii=False, separators=(",", ":"))
            + "\n【校验错误】\n" + json.dumps(issues, ensure_ascii=False)
            + "\n【当前世界候选；请在它上面局部修改】\n"
            + json.dumps(asdict(world), ensure_ascii=False, separators=(",", ":")),
            temperature=0.0,
            max_tokens=config_int("scenario", "repair_max_tokens", 4096, minimum=512, maximum=10000),
            validate_payload=lambda candidate: apply_scenario_patch(package, candidate),
        )
        candidate = apply_scenario_patch(package, patch).world
        candidate, _ = _repair_vote_contract(candidate, user_prompt)
        candidate, _ = _repair_world_phase_beats(candidate)
        candidate, _ = _prune_unreachable_noop_rules(candidate)
        candidate, _ = _prune_shadow_vote_wins(candidate)
        candidate, _ = _repair_ambiguous_phase_transitions(candidate)
        candidate, _ = _repair_mechanical_world_fields(candidate)
        _ensure_short_input_opening_room(candidate, brief, user_prompt)
        candidate_issues = validate_world_structure(candidate)
        improved = _semantic_progress(
            issues, candidate_issues, fingerprint(asdict(candidate)) != fingerprint(asdict(world)),
        )
        ledger["attempts"] += 1
        ledger["no_progress"] = 0 if improved else ledger["no_progress"] + 1
        ledger["history"].append({
            "before_count": len(issues), "after_count": len(candidate_issues),
            "accepted": improved,
            "new_components": sorted({_issue_anchor(item) for item in candidate_issues}
                                     - {_issue_anchor(item) for item in issues})[:20],
        })
        ledger["history"] = ledger["history"][-6:]
        if improved:
            world, issues = candidate, candidate_issues
            ledger["candidate"] = asdict(world)
        control.progress(candidate_issue_count=len(candidate_issues), candidate_accepted=improved,
                         reason="世界候选错误减少，已保存" if improved else "补丁未减少错误或引入新故障，保留原候选")
        control.save_recovery()
        if issues:
            check_stalled()
    if issues:
        check_stalled()
        raise ScenarioValidationError(issues)
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
        f"{CHARACTER_DESIGN_PROMPT}\n\n{prompt_contract('scenario')}\n\n{CHARACTER_JSON_INSTRUCTION}\n\n{RUNTIME_GENERATION_GUIDANCE}",
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
节点若标为 completion_mode=state，completion_conditions 不能是空数组，也不能填 all_eligible_acted/all_active_voted/phase_advanced 等阶段推进条件。若报错只涉及这种节点，且没有可声明的世界/实体/角色状态结果，应同时将该节点的 completion_mode 改成 narrative、清空无效条件，并把 resolution_signals 改为可由已提交角色行动核验的证据；不能只删掉条件而保留 state。纯粹的开场、投票、结算等自动流程不是用户明确要求的故事里程碑时，也可以移除对应 beat_specs，并同步修正 target_beats 与 prerequisites；不要把不存在的主持人行动写成证据。
所有 covered_constraint_ids 必须真实对应其落实位置。公共世界仍不得包含导演秘密。
如果错误涉及规则运行时，只修改报错涉及的 phase_specs、rules、state_schema、entities 或 termination_rules，以及必要关联。termination_rules.kind 只支持 faction_eliminated、faction_parity、world_equals、entity_equals、all_goals_completed、all_active_at_location、all_of、any_of、manual；禁止创造 ai_win、human_win、entity_state_equals 等新 kind。all_of/any_of 必须有非空 conditions。阶段只允许与当前设定相符的行动；强制投票阶段可以只有 vote，pass 不计入 all_active_voted；manual 且有 next_phase 时必须有可执行 set_phase 规则。
严格保留 world.execution_version。版本1才有 move/vote/inspect/protect/eliminate/poison/heal 的隐式内置效果。版本2行动名称没有隐式结果，必须填写白名单 effects；effect_mode=rule/ability/stack 控制来源，不得为了修复把正常 effects 清空。legacy_tabletop 仅可显式用于用户确实要求的传统桌游机制，不能套到会议、医疗、战斗等场景。consume_ability 由引擎处理，不得重复填写。禁止脚本、条件表达式和未支持操作，不要退回自然语言规则冒充可执行字段。"""
    control = CURRENT_BUILD.get()
    history = (control.recovery.get("semantic", {}).get("history", [])[-3:] if control else [])
    payload = _invoke_json(
        repair_prompt + "\n\n" + RUNTIME_GENERATION_GUIDANCE,
        "【用户原始输入】\n"
        f"{user_prompt}\n\n"
        "【不可修改的约束账本】\n"
        f"{json.dumps(package.brief.to_dict(), ensure_ascii=False, separators=(',', ':'))}\n\n"
        "【校验错误】\n"
        f"{json.dumps(issues, ensure_ascii=False)}\n\n"
        "【先前修复结果（不要重复无进展方案）】\n"
        f"{json.dumps(history, ensure_ascii=False)}\n\n"
        "【待修复结果】\n"
        f"{json.dumps(package.to_dict(), ensure_ascii=False, separators=(',', ':'))}",
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
    package.world, repaired_votes = _repair_vote_contract(package.world, user_prompt)
    package.world, removed_vote_wins = _prune_shadow_vote_wins(package.world)
    package.world, repaired_transitions = _repair_ambiguous_phase_transitions(package.world)
    package.world, _ = _repair_mechanical_world_fields(package.world)
    _ensure_short_input_opening_room(package.world, package.brief, user_prompt)
    _repair_generic_relationship_targets(package, user_prompt)
    save(package)
    issues = validate_scenario_package(package, user_prompt=user_prompt)
    ledger = control.recovery.setdefault("semantic", {"attempts": 0, "no_progress": 0, "history": []})
    attempts = 0
    if changes:
        control.progress(reason="已本地纠正明确的字段错位，无需请求模型",
                         patched_paths=[item["path"] for item in changes])
    if repaired_votes:
        control.progress(reason="已修正结构化投票的阶段与弃权契约",
                         local_repaired_vote_contract=repaired_votes)
    if removed_vote_wins:
        control.progress(reason="已移除与结束规则重复且不记录选票的伪投票规则",
                         local_removed_vote_wins=removed_vote_wins)
    if repaired_transitions:
        control.progress(reason="已纠正普通发言误触发阶段切换及角色占用公告阶段",
                         local_repaired_transitions=repaired_transitions)
    def check_stalled():
        if ledger["no_progress"] >= 2 or ledger["attempts"] >= config_int(
                "scenario", "semantic_total_attempts", 6, minimum=1, maximum=12):
            summary = "；".join(issues[:3])[:360]
            raise SceneChatError("scenario_repair_stalled",
                                 "一致性修复未取得进展或已达累计上限，已保留较好的检查点。"
                                 + (f"尚未解决：{summary}。" if summary else "")
                                 + "请修改设定或模型后重新生成；重复继续不会再调用修复模型。",
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
        candidate.world, _ = _repair_vote_contract(candidate.world, user_prompt)
        candidate.world, _ = _prune_shadow_vote_wins(candidate.world)
        candidate.world, _ = _repair_ambiguous_phase_transitions(candidate.world)
        candidate.world, _ = _repair_mechanical_world_fields(candidate.world)
        _ensure_short_input_opening_room(candidate.world, candidate.brief, user_prompt)
        _repair_generic_relationship_targets(candidate, user_prompt)
        candidate_issues = validate_scenario_package(candidate, user_prompt=user_prompt)
        improved = _semantic_progress(
            issues, candidate_issues,
            fingerprint(candidate.to_dict()) != fingerprint(package.to_dict()),
        )
        ledger["attempts"] += 1
        attempts += 1
        ledger["no_progress"] = 0 if improved else ledger["no_progress"] + 1
        ledger["history"].append({"before_issues": issues[:30], "after_issues": candidate_issues[:30],
                                  "before_count": len(issues), "after_count": len(candidate_issues),
                                  "new_components": sorted({_issue_anchor(item) for item in candidate_issues}
                                                           - {_issue_anchor(item) for item in issues})[:20],
                                  "accepted": improved,
                                  "patched_paths": control.details.get("patched_paths", [])})
        ledger["history"] = ledger["history"][-6:]
        if improved:
            package, issues = candidate, candidate_issues
            save(package)
        control.progress(candidate_issue_count=len(candidate_issues),
                         candidate_accepted=improved,
                         reason=("局部修复已减少错误且未新增故障组件" if improved else
                                 "修复没有减少故障组件，或引入了新的故障组件；原检查点保持不变"))
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
