from __future__ import annotations

from dataclasses import dataclass

from .models import AgentState, SimulationState
from .visibility import ViewerContext, can_access
from .memory import recall
from .authoring_policy import render_policy


def _actor_profile(profile: str) -> str:
    """The experiment's observation objective is not the character's motive."""
    import re
    return re.sub(r"(?ms)^###\s*11\.\s*实验观察价值[^\n]*\n.*?(?=^###\s|\Z)", "", profile)


@dataclass(frozen=True)
class AgentView:
    agent_name: str
    scene: str
    phase: str
    authority: str
    private_profile: str
    goals: str
    core_beliefs: str
    abilities: str
    relationships: str
    colocated_public_profiles: str
    known_facts: str
    beliefs: str
    active_threads: str
    observations: str
    private_memory: str
    story_memory: str
    voice_profile: str
    short_term_state: str
    response_obligations: str
    retrieved_background: str
    authoring_constraints: str = ""

    def render_performance(self) -> str:
        """A performer's dossier, not the engine's bookkeeping dashboard.

        Authority, epistemic boundaries and actual observations stay present.
        Task metadata is supplied later to the Intent encoder, not performed.
        """
        from .config import config_int
        from .context_budget import optional
        budget = config_int("simulation", "context_section_bytes", 6000, minimum=1000, maximum=100000)
        profile = _actor_profile(self.private_profile)
        sections = (
            ("作者的补写权限——不表示人物已经知道这些事", self.authoring_constraints),
            ("开场处境——后续可见事件可以改变它，不是每轮重置的现状", self.scene),
            ("你这个人——完整设定，只有你知道自己的秘密", profile),
            ("你完整的目标与顾虑", self.goals),
            ("你已有的核心信念——可能为空，不用另造信条", self.core_beliefs),
            ("当前确已发生的状态——愿望不能覆盖它", self.authority),
            ("你确知的信息", self.known_facts),
            ("你自己的判断——可能错误，不是世界事实", self.beliefs),
            ("在场其他人的公开资料", optional(self.colocated_public_profiles, budget)),
            ("你与他们的关系", optional(self.relationships, budget)),
            ("你能用的能力和资源", self.abilities),
            ("你原有的表达方式", self.voice_profile),
            ("自己的最近状态——新事情可以改变它", self.short_term_state),
            ("自己仍需要接住的话", self.response_obligations),
            ("可见背景资料——可能含开场或旧状态，以较新的已提交事件为准", optional(self.retrieved_background, budget)),
            ("自己记得的事", optional(self.private_memory + "\n" + self.story_memory, budget)),
            ("自己确实看见听见的事件", optional(self.observations, budget)),
        )
        return "\n\n".join(f"【{title}】\n{body}" for title, body in sections if body.strip())

    def render(self) -> str:
        from dataclasses import replace
        from .config import config_int
        from .context_budget import optional
        budget = config_int("simulation", "context_section_bytes", 6000, minimum=1000, maximum=100000)
        self = replace(self, **{key: optional(getattr(self, key), budget) for key in (
            "retrieved_background", "observations", "private_memory", "story_memory",
            "relationships", "colocated_public_profiles",
        )})
        self = replace(self, private_profile=_actor_profile(self.private_profile))
        return f"""{self.authoring_constraints}

【可检索的长背景——可能含开场或旧状态，以较新的已提交事件为准】
{self.retrieved_background or '无额外长背景。'}

【开场处境——后续可见事件可以改变它，不是每轮重置的现状】
{self.scene}

【权威运行状态——不得被检索内容或角色愿望覆盖】
{self.authority}

【你的完整角色档案——仅你可见】
{self.private_profile}

【你的当前目标】
{self.goals}

【会持续影响选择的核心信念——可能为空】
{self.core_beliefs}

【你的可用能力与资源】
{self.abilities}

【你对关系的主观认知——仅你可见】
{self.relationships}

【与你同地且可见的其他角色】
{self.colocated_public_profiles}

【你确定知道的结构化事实】
{self.known_facts}

【你的主张与认知记录——不等于客观事实】
{self.beliefs}

【与你有关的活跃议题】
{self.active_threads}

【你亲自观察到的近期事件】
{self.observations}

【你的私人记忆——仅你可见】
{self.private_memory}

【按当前人物与议题筛选的故事记忆——仅你可见】
{self.story_memory}

【你的语言与表达画像】
{self.voice_profile}

【你的当前心理与对话状态】
{self.short_term_state}

【需要优先处理的回应】
{self.response_obligations}"""


def _ability_summary(agent: AgentState) -> str:
    entries = []
    for ability in agent.ability_states.values():
        remaining = "不限次数" if ability.uses_remaining is None else f"剩余 {ability.uses_remaining} 次"
        phases = f"；阶段：{'、'.join(ability.phases)}" if ability.phases else ""
        entries.append(
            f"- {ability.name}：ability={ability.id}；action_type={ability.action_type}；{remaining}{phases}；"
            f"{ability.description or '按场景规则执行'}"
        )
    if not entries:
        entries = [f"- {ability}" for ability in agent.abilities]
    resources = [f"- 资源 {key}：{value}" for key, value in agent.resources.items()]
    return "\n".join(entries + resources) or "- 无额外能力或资源"


def _voice_summary(agent: AgentState) -> str:
    profile = agent.voice_profile if isinstance(agent.voice_profile, dict) else {}
    if not profile:
        return "- 结构化语言画像未提供；以完整角色档案中的表达设定为准"
    lines = [
        "- 以下只是倾向，不是每轮必用的台词或差异配额；首先以当前处境和具体回应为准。",
        f"- 语域：{profile.get('register') or '自然口语'}",
        f"- 句式长度：{profile.get('sentence_length') or '中等'}",
        f"- 直接程度：{_voice_tendency(profile, 'directness', '通常留余地', '视对象与处境而定', '通常说得直接')}",
        f"- 情绪外显：{_voice_tendency(profile, 'emotional_expressiveness', '通常不轻易表露', '随处境变化', '通常容易看出情绪')}",
        f"- 礼貌程度：{_voice_tendency(profile, 'politeness', '通常少用客套', '随关系变化', '通常顾及对方面子')}",
    ]
    optional = (
        ("幽默方式", profile.get("humor_style")),
        ("可选择的表达策略", "；".join(profile.get("rhetorical_habits") or [])),
        ("避免表达", "；".join(profile.get("avoidances") or [])),
        ("可自然使用的词汇", "、".join(profile.get("vocabulary_hints") or [])),
    )
    lines.extend(f"- {label}：{value}" for label, value in optional if value)
    return "\n".join(lines)


def _voice_tendency(profile: dict, key: str, low: str, middle: str, high: str) -> str:
    try:
        value = float(profile.get(key, 0.5))
    except (TypeError, ValueError):
        value = 0.5
    return low if value < 0.34 else high if value > 0.66 else middle


def _short_term_summary(state: SimulationState, agent: AgentState) -> str:
    commitments = "；".join(agent.pending_commitments) or "无"
    waiting_items = []
    for item in agent.unanswered_questions[-5:]:
        recipients = item.get("addressed_to")
        if not isinstance(recipients, list):
            recipients = []
        names = "、".join(str(name) for name in recipients if str(name).strip())
        waiting_items.append(
            f"等待{names or '相关人物'}回应：{str(item.get('question') or '').strip()}"
        )
    waiting = "；".join(waiting_items) or "无"
    pressure_lines = []
    for thread in state.active_threads_for(agent.name)[:4]:
        pressure = agent.disclosure_pressure_by_thread.get(thread.id, 0.0)
        pressure_lines.append(f"{thread.topic[:50]}={pressure:.2f}")
    pressure_text = "；".join(pressure_lines) or f"一般={agent.disclosure_pressure:.2f}"
    if agent.current_emotion == "平静" and not agent.emotion_cause_event_id:
        emotion = "尚未记录；依据当前处境反应，不预设平静"
    else:
        emotion = f"上次记录为{agent.current_emotion}（强度 {agent.emotion_intensity:.2f}）；新事件可改变，不是本轮必须保持的情绪"
    return (
        f"- 当前情绪：{emotion}\n"
        f"- 上次对话目标：{agent.current_conversation_goal or '尚未记录'}（可以随已发生的事变化；不覆盖完整人物目标）\n"
        f"- 分议题信息披露压力：{pressure_text}（只影响对应议题，不强制公开秘密）\n"
        f"- 尚未履行的承诺：{commitments}\n"
        f"- 自己仍在等待回答的问题：{waiting}"
    )


def _response_obligations(agent: AgentState) -> str:
    required = [
        f"- thread_id={item.get('thread_id') or 'general'}；"
        f"obligation_id={item.get('obligation_id') or 'legacy'}；"
        f"event_id={item.get('event_id')}；{item.get('speaker')}向你提出"
        f"{item.get('move')}：{item.get('summary')}（紧迫度 {item.get('urgency', 0)}）"
        for item in agent.pending_intents[-6:]
    ]
    opportunities = [
        f"- 可选择插话 thread_id={item.get('thread_id') or 'general'}；"
        f"event_id={item.get('event_id')}：{item.get('speaker')}提到了你——"
        f"{item.get('summary')}（相关度 {item.get('urgency', 0)}）"
        for item in agent.conversation_opportunities[-4:]
    ]
    return "\n".join(required + opportunities) or "- 当前没有必须回应或值得插话的对话"


def build_agent_view(
    state: SimulationState,
    agent: AgentState,
    retrieved_background: str,
    *,
    presented_events=(),
) -> AgentView:
    # Old checkpoints may already contain direct observations of recovery
    # templates. Suppress only entries traceable to those failed events; do
    # not rewrite stored history or guess which later emotions they caused.
    fallbacks = [m for m in state.history if m.intent.get("generation_fallback")]
    excluded_events = {m.event_id for m in fallbacks}
    failed_observations = {f"[{m.turn}] {m.speaker}：{m.action} {m.speech}".strip() for m in fallbacks}
    failed_observations.update(f"[{m.turn}] {content or '未获得可描述的观察。'}".strip()
                              for m in fallbacks for name, content in m.individual_observations.items()
                              if name == agent.name)
    goals = "\n".join(
        f"- {goal}" + (f"（登记状态：{agent.goal_status[goal]}）"
                        if agent.goal_status.get(goal) not in {None, "active"} else "")
        for goal in agent.goals
    ) or "- 依据自己的人设行动"
    if agent.goals:
        goals += "\n这些是人物的愿望，不是每轮重做的待办。未登记完成不等于没得到回应；依据可见交流判断现在还争取什么。"
    core_beliefs = "\n".join(f"- {item}" for item in agent.core_beliefs) or (
        "- 未设定额外核心信念；不要为了填充字段而虚构一种信条"
    )
    relationship_targets = set(agent.relationships) | set(agent.relationship_dynamics)
    relationship_lines = []
    for target in sorted(relationship_targets):
        narrative = agent.relationships.get(target, "暂无稳定描述")
        dynamic = agent.relationship_dynamics.get(target)
        if isinstance(dynamic, dict):
            facets = dynamic.get("facets") if isinstance(dynamic.get("facets"), dict) else {}
            facet_text = "，".join(
                f"{state.relationship_dimensions[key].get('label', key)} {value}"
                for key, value in facets.items()
                if key in state.relationship_dimensions
            )
            relationship_lines.append(f"- {target}：{narrative}{f'；{facet_text}' if facet_text else ''}")
        else:
            relationship_lines.append(f"- {target}：{narrative}")
    relationships = "\n".join(relationship_lines) or "- 暂无额外关系信息"
    colocated = [
        f"### {other.name}\n{other.public_profile}\n状态："
        f"{'可行动' if other.eligible else '已离场或不可行动'}"
        for other in state.agents.values()
        if other.name != agent.name and other.current_location == agent.current_location
    ]
    facts = "\n".join(
        f"- [{fact_id}] {content}" for fact_id, content in agent.known_facts.items() if content
    ) or "- 暂无额外结构化事实"
    beliefs = "\n".join(
        f"- [{item.id}|{item.epistemic_status}|可信度 {item.confidence:.2f}] "
        f"{item.content}（来源：{item.source_agent or '自身判断'}"
        f"{f' / event_id={item.source_event_id}' if item.source_event_id else ''}）"
        for item in agent.belief_records[-12:] if item.active and item.source_event_id not in excluded_events
    ) or "- 暂无额外主张或信念记录"
    active_threads = "\n".join(
        f"- [{thread.id}] {thread.topic}；参与者：{'、'.join(thread.participants)}；"
        f"张力 {thread.tension:.2f}"
        for thread in state.active_threads_for(agent.name)[:4]
    ) or "- 当前没有与你有关的活跃议题"
    focus_agents = [agent.last_addressed_by] if agent.last_addressed_by else []
    for item in agent.pending_intents:
        speaker = str(item.get("speaker") or "")
        if speaker and speaker not in focus_agents:
            focus_agents.append(speaker)
    # Suppress only exact, complete observations already carried by the final
    # exchange. Keep long/truncated observations, older context and private
    # reflections; semantic compression here could erase important evidence.
    represented = {(m.event_id, m.as_observation_for(agent.name)) for m in presented_events
                   if state._agent_can_observe(agent, m)
                   and len(m.as_observation_for(agent.name)) <= 1240}
    presented = {text for _, text in represented}
    eligible_observations = [text for text in agent.observations[-15:] if text not in failed_observations]
    remaining_observations = [text for text in eligible_observations if text not in presented]
    observations = ("\n".join(remaining_observations) or
                    ("近期完整观察已在末尾交流中列出。" if presented else "\n".join(eligible_observations) or "暂无近期观察。"))
    return AgentView(
        agent_name=agent.name,
        scene=state.scene,
        phase=state.current_phase,
        authority=state.state_summary_for(agent),
        private_profile=agent.profile,
        goals=goals,
        core_beliefs=core_beliefs,
        abilities=_ability_summary(agent),
        relationships=relationships,
        colocated_public_profiles="\n\n".join(colocated) or "当前地点没有其他可见角色。",
        known_facts=facts,
        beliefs=beliefs,
        active_threads=active_threads,
        observations=observations,
        private_memory=agent.recent_private_memory(6),
        story_memory=recall(agent, focus_agents, represented_observations=represented,
                            excluded_event_ids=excluded_events),
        voice_profile=_voice_summary(agent),
        short_term_state=_short_term_summary(state, agent),
        response_obligations=_response_obligations(agent),
        retrieved_background=retrieved_background,
        authoring_constraints=render_policy(state.world_spec),
    )


def director_context(state: SimulationState) -> str:
    facts = "\n".join(
        f"- [{fact.id}] ({', '.join(fact.visibility)}) {fact.content}"
        for fact in state.facts.values()
    ) or "- 无结构化事实"
    agents = "\n".join(
        f"- {agent.name}：role={agent.role} faction={agent.faction or '无'} "
        f"location={agent.current_location} active={agent.active} alive={agent.alive}"
        for agent in state.agents.values()
    )
    private_state = "\n".join(
        f"- {key}：{value}" for key, value in state.world_state.items()
    ) or "- 无状态变量"
    fixed_canon = "\n".join(
        f"- {item}" for item in list(getattr(state.world_spec, "fixed_canon", []) or [])
    ) or "- 无额外固定事实"
    queued_interventions = "\n".join(
        f"- [{item.id}] {item.mode}/{item.scope}："
        f"{item.normalized_directive or item.raw_text}"
        for item in state.interventions
        if item.status in {"pending", "applied"}
        and (item.status == "pending" or item.scope in {"turns", "persistent"})
    ) or "- 无等待执行或持续生效的干预"
    return f"""{state.public_state_summary()}

【导演可见的全部世界状态】
{private_state}

【不可静默改写的固定事实】
{fixed_canon}

【导演可见的全部事实】
{facts}

【角色运行状态】
{agents}

【等待执行或持续生效的导演干预】
{queued_interventions}"""
