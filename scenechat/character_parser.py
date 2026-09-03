import re
from typing import Dict, List

from .models import AgentState


CHARACTER_HEADER_RE = re.compile(r"(?=^##\s*角色\s*\d+\s*$)", re.MULTILINE)
SECTION_RE = re.compile(
    r"^###\s*(\d+)\.\s*([^\n]+)\n(.*?)(?=^###\s*\d+\.|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _clean_value(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^\*\*(.*?)\*\*$", r"\1", value)
    return value.strip()


def _sections(profile: str) -> Dict[int, tuple[str, str]]:
    return {
        int(number): (_clean_value(title), _clean_value(value))
        for number, title, value in SECTION_RE.findall(profile)
    }


def split_character_profiles(markdown: str) -> List[str]:
    """Split the generated character document into complete role dossiers."""
    chunks = CHARACTER_HEADER_RE.split(markdown)
    return [chunk.strip() for chunk in chunks if chunk.lstrip().startswith("## 角色")]


def parse_character_agents(markdown: str) -> List[AgentState]:
    """Convert Markdown dossiers into isolated, structured agent state objects."""
    agents: List[AgentState] = []

    for profile in split_character_profiles(markdown):
        sections = _sections(profile)
        name = sections.get(1, ("", ""))[1].splitlines()[0].strip()
        if not name:
            continue

        public_parts = []
        for number in range(1, 5):
            title, value = sections.get(number, ("", ""))
            if value:
                public_parts.append(f"{title or f'公开信息 {number}'}：{value}")

        goal = sections.get(7, ("", ""))[1]
        goals = [goal] if goal else []

        relation_number = 10 if "认知" in sections.get(10, ("", ""))[0] else 11
        relation_value = sections.get(relation_number, ("", ""))[1]
        relationships = {"初始关系认知": relation_value} if relation_value else {}

        private_memory = []
        knowledge_title, knowledge_value = sections.get(8, ("", ""))
        if knowledge_value and ("知识" in knowledge_title or "信息边界" in knowledge_title):
            private_memory.append(f"我的初始知识边界：{knowledge_value}")
        belief_match = re.search(
            r"^###\s*7\.1\s*[^\n]*\n(.*?)(?=^###\s*\d+\.|\Z)",
            profile,
            re.MULTILINE | re.DOTALL,
        )
        belief_text = _clean_value(belief_match.group(1)) if belief_match else ""
        core_beliefs = [] if (
            not belief_text or belief_text == "未设定额外核心信念"
        ) else [belief_text]

        agents.append(
            AgentState(
                name=name,
                profile=profile,
                public_profile="\n".join(public_parts),
                goals=goals,
                private_memory=private_memory,
                relationships=relationships,
                core_beliefs=core_beliefs,
            )
        )

    return agents
