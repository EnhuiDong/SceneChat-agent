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
{self.private_memory}"""


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


def build_agent_view(
    state: SimulationState,
    agent: AgentState,
    retrieved_background: str,
) -> AgentView:
    goals = "\n".join(
        f"- {goal}（{agent.goal_status.get(goal, 'active')}）" for goal in agent.goals
    ) or "- 依据自己的人设行动"
    relationships = "\n".join(
        f"- {key}：{value}" for key, value in agent.relationships.items()
    ) or "- 暂无额外关系信息"
    colocated = [
        f"### {other.name}\n{other.public_profile}\n状态："
        f"{'可行动' if other.eligible else '已离场或不可行动'}"
        for other in state.agents.values()
        if other.name != agent.name and other.current_location == agent.current_location
    ]
    facts = "\n".join(
        f"- [{fact_id}] {content}" for fact_id, content in agent.known_facts.items() if content
    ) or "- 暂无额外结构化事实"
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
        private_memory=agent.recent_private_memory(),
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
    return f"""{state.public_state_summary()}

【导演可见的全部世界状态】
{private_state}

【导演可见的全部事实】
{facts}

【角色运行状态】
{agents}"""
