import history
import uuid
from scenechat.errors import SceneChatError, stage_error
from scenechat.generation import generate_scenario_package
from scenechat.knowledge import requires_vector_index
from scenechat.preflight import (
    validate_embedding_model_availability,
    validate_generation_model_availability,
)
from scenechat.storage import save_experiment_documents

def main():
    print("欢迎来到社会模拟实验设定生成器")
    print("-" * 40)

    user_prompt = input("请输入你想要什么样的社会实验：\n> ").strip()

    if not user_prompt:
        print("输入不能为空，请重新运行程序。")
        return

    try:
        print("\n正在检查生成模型...")
        validate_generation_model_availability()
        print("生成模型检查通过。")
    except SceneChatError as error:
        print(f"\n{error.public_message}")
        return
    except Exception as exc:
        print(f"\n{stage_error('preflight', exc).public_message}")
        return

    try:
        print("\n正在理解约束并生成结构化场景，请稍等...\n")
        package = generate_scenario_package(user_prompt)
        worldview = package.worldview_markdown

        print("=" * 60)
        print("【生成的世界观】")
        print("=" * 60)
        print(worldview)

        characters = package.characters_markdown

        if requires_vector_index(
            package.public_worldview_markdown,
            characters,
            package.world.director_notes_markdown,
            package.world.facts,
        ):
            print("\n检测到长背景，正在检查向量模型...")
            validate_embedding_model_availability()

        print("=" * 60)
        print("【生成的角色设定】")
        print("=" * 60)
        print(characters)

        experiment_id = uuid.uuid4().hex
        save_experiment_documents(
            experiment_id,
            worldview,
            characters,
            scenario_payload=package.to_dict(),
        )

    except Exception as exc:
        # Keep CLI feedback consistent with the Web API and never print raw
        # provider payloads as the primary user-facing error.
        stage = "scenario_generation"
        print(f"\n{stage_error(stage, exc).public_message}")
        return

    history.main(worldview, characters, scenario_package=package)


if __name__ == "__main__":
    main()
