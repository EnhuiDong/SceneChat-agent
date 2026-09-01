from __future__ import annotations

from dataclasses import dataclass

from .models import AgentState, SimulationState
from .visibility import ViewerContext, can_access


@dataclass(frozen=True)
class AgentView:
    agent_name: str
    scene: str
    phase: str
    authority: str
    private_profile: str
    goals: str
    abilities: str
    relationships: str
    colocated_public_profiles: str
    known_facts: str
    observations: str
    private_memory: str
    story_memory: str
    voice_profile: str
    short_term_state: str
    response_obligations: str
    retrieved_background: str

    def render(self) -> str:
        return f"""【可检索的长背景】
{self.retrieved_background or '无额外长背景。'}

【当前场景】
{self.scene}

【权威运行状态——不得被检索内容或角色愿望覆盖】
{self.authority}

【你的完整角色档案——仅你可见】
{self.private_profile}

【你的当前目标】
{self.goals}

【你的可用能力与资源】
{self.abilities}

【你对关系的主观认知——仅你可见】
{self.relationships}

【与你同地且可见的其他角色】
{self.colocated_public_profiles}

【你确定知道的结构化事实】
{self.known_facts}

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
            f"- {ability.name} [{ability.id}/{ability.action_type}]：{remaining}{phases}；"
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
        f"- 语域：{profile.get('register') or '自然口语'}",
        f"- 句式长度：{profile.get('sentence_length') or '中等'}",
        f"- 直接程度：{profile.get('directness', 0.5)}",
        f"- 情绪外显：{profile.get('emotional_expressiveness', 0.5)}",
        f"- 礼貌程度：{profile.get('politeness', 0.5)}",
    ]
    optional = (
        ("幽默方式", profile.get("humor_style")),
        ("表达策略", "；".join(profile.get("rhetorical_habits") or [])),
        ("避免表达", "；".join(profile.get("avoidances") or [])),
        ("可自然使用的词汇", "、".join(profile.get("vocabulary_hints") or [])),
    )
    lines.extend(f"- {label}：{value}" for label, value in optional if value)
    return "\n".join(lines)


def _short_term_summary(agent: AgentState) -> str:
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
    return (
        f"- 当前情绪：{agent.current_emotion}（强度 {agent.emotion_intensity:.2f}）\n"
        f"- 当前对话目标：{agent.current_conversation_goal or '依据长期目标判断'}\n"
        f"- 信息披露压力：{agent.disclosure_pressure:.2f}（越高越难继续完全回避，但仍由人设决定如何回应）\n"
        f"- 尚未履行的承诺：{commitments}\n"
        f"- 自己仍在等待回答的问题：{waiting}"
    )


def _response_obligations(agent: AgentState) -> str:
    required = [
        f"- event_id={item.get('event_id')}；{item.get('speaker')}向你提出"
        f"{item.get('move')}：{item.get('summary')}（紧迫度 {item.get('urgency', 0)}）"
        for item in agent.pending_intents[-6:]
    ]
    opportunities = [
        f"- 可选择插话 event_id={item.get('event_id')}：{item.get('speaker')}提到了你——"
        f"{item.get('summary')}（相关度 {item.get('urgency', 0)}）"
        for item in agent.conversation_opportunities[-4:]
    ]
    return "\n".join(required + opportunities) or "- 当前没有必须回应或值得插话的对话"


def build_agent_view(
    state: SimulationState,
    agent: AgentState,
    retrieved_background: str,
) -> AgentView:
    goals = "\n".join(
        f"- {goal}（{agent.goal_status.get(goal, 'active')}）" for goal in agent.goals
    ) or "- 依据自己的人设行动"
    relationship_targets = set(agent.relationships) | set(agent.relationship_dynamics)
    relationship_lines = []
    for target in sorted(relationship_targets):
        narrative = agent.relationships.get(target, "暂无稳定描述")
        dynamic = agent.relationship_dynamics.get(target)
        if isinstance(dynamic, dict):
            relationship_lines.append(
                f"- {target}：{narrative}；信任 {dynamic.get('trust', 0.5)}，"
                f"怀疑 {dynamic.get('suspicion', 0.5)}，亲近 {dynamic.get('affinity', 0.5)}"
            )
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
    focus_agents = [agent.last_addressed_by] if agent.last_addressed_by else []
    for item in agent.pending_intents:
        speaker = str(item.get("speaker") or "")
        if speaker and speaker not in focus_agents:
            focus_agents.append(speaker)
    return AgentView(
        agent_name=agent.name,
        scene=state.scene,
        phase=state.current_phase,
        authority=state.state_summary_for(agent),
        private_profile=agent.profile,
        goals=goals,
        abilities=_ability_summary(agent),
        relationships=relationships,
        colocated_public_profiles="\n\n".join(colocated) or "当前地点没有其他可见角色。",
        known_facts=facts,
        observations=agent.recent_observations(),
        private_memory=agent.recent_private_memory(6),
        story_memory=agent.layered_memory(focus_agents),
        voice_profile=_voice_summary(agent),
        short_term_state=_short_term_summary(agent),
        response_obligations=_response_obligations(agent),
        retrieved_background=retrieved_background,
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
