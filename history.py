"""Backward-compatible CLI facade for the SceneChat simulation core."""

from pathlib import Path
from typing import Optional
import uuid

from dotenv import load_dotenv

from scenechat.character_parser import parse_character_agents
from scenechat.errors import stage_error
from scenechat.knowledge import (
    ExperimentKnowledgeBase,
    build_experiment_knowledge_base,
)
from scenechat.models import AgentState, Message, SimulationState
from scenechat.providers import get_embedding_model, get_simulation_llm
from scenechat.simulation import simulate_next_event, simulate_next_turn
from scenechat.scenario import ScenarioPackage, agents_from_character_specs


load_dotenv()

DEFAULT_BATCH_SIZE = 5
MAX_GLOBAL_HISTORY = 15


def get_llm():
    """Keep the original public helper while lazily creating the provider client."""
    return get_simulation_llm()


def get_embed_model():
    return get_embedding_model()


def get_index(
    worldview: Optional[str] = None,
    characters: Optional[str] = None,
    experiment_id: Optional[str] = None,
    director_notes: str = "",
    facts=None,
) -> ExperimentKnowledgeBase:
    """Build an isolated experiment knowledge base.

    When called without arguments, this retains the original CLI behavior of
    loading ``data/world.md`` and ``data/character.md``.
    """
    if worldview is None:
        world_path = Path("data/world.md")
        if not world_path.exists():
            raise FileNotFoundError("data/world.md is not found")
        worldview = world_path.read_text(encoding="utf-8")

    if characters is None:
        character_path = Path("data/character.md")
        if not character_path.exists():
            raise FileNotFoundError("data/character.md is not found")
        characters = character_path.read_text(encoding="utf-8")

    return build_experiment_knowledge_base(
        worldview,
        characters,
        experiment_id or uuid.uuid4().hex,
        director_notes=director_notes,
        facts=facts,
    )


def main(
    worldview: Optional[str] = None,
    characters: Optional[str] = None,
    scenario_package: Optional[ScenarioPackage] = None,
) -> None:
    initial_scene = input(
        "请输入初始场景（留空则使用根据题材生成的开场）："
    ).strip()
    if not initial_scene:
        initial_scene = (
            scenario_package.world.opening_scene
            if scenario_package is not None
            else "所有参与者已经到场，用户设定的事件即将开始。"
        )

    if worldview is None:
        worldview = Path("data/world.md").read_text(encoding="utf-8")
    if characters is None:
        characters = Path("data/character.md").read_text(encoding="utf-8")

    experiment_id = uuid.uuid4().hex
    try:
        if scenario_package is not None:
            knowledge_base = get_index(
                scenario_package.public_worldview_markdown,
                characters,
                experiment_id,
                director_notes=scenario_package.world.director_notes_markdown,
                facts=scenario_package.world.facts,
            )
            agents = agents_from_character_specs(scenario_package.characters)
            state = SimulationState(initial_scene, agents, world_spec=scenario_package.world)
        else:
            knowledge_base = get_index(worldview, characters, experiment_id)
            state = SimulationState(initial_scene, parse_character_agents(characters))
    except Exception as exc:
        print(f"\n{stage_error('index', exc).public_message}")
        return
    print(f"\n场景设定：{initial_scene}\n")
    simulation_llm = get_llm()

    while True:
        try:
            raw_count = input(
                f"\n输入推演轮数（默认 {DEFAULT_BATCH_SIZE}，'quit' 退出）："
            ).strip()
            if raw_count.lower() == "quit":
                break
            count = int(raw_count) if raw_count else DEFAULT_BATCH_SIZE

            print(f"\n开始推演 {count} 轮...\n")
            for _ in range(max(0, count)):
                if state.ended:
                    print(f"场景已经结束：{state.end_reason}")
                    break
                msg = simulate_next_event(state, knowledge_base, llm=simulation_llm)
                if msg is None:
                    print("LLM 输出解析失败，跳过本轮。")
                    continue
                state.add_message(msg)
                print(f"[{msg.turn}] {msg.speaker}: {msg.action} {msg.speech}")

            print(f"\n本批完成，总轮数：{state.turn_count}")
        except ValueError:
            print("请输入有效数字。")
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"Error: {exc}")

    if input("\n保存日志到 simulation_log.txt? (y/n): ").lower() == "y":
        with open("simulation_log.txt", "w", encoding="utf-8") as log_file:
            log_file.write(f"场景: {state.scene}\n\n")
            for msg in state.history:
                log_file.write(msg.as_observation() + "\n")
        print("已保存。")

    print("推演结束。")


__all__ = [
    "AgentState",
    "Message",
    "SimulationState",
    "get_embed_model",
    "get_index",
    "get_llm",
    "simulate_next_turn",
    "simulate_next_event",
]


if __name__ == "__main__":
    main()
