import history
from World import generate_worldview
from Character import generate_characters
from history import main

def main():
    print("欢迎来到社会模拟实验设定生成器")
    print("-" * 40)

    user_prompt = input("请输入你想要什么样的社会实验：\n> ").strip()

    if not user_prompt:
        print("输入不能为空，请重新运行程序。")
        return

    try:
        print("\n正在生成世界观，请稍等...\n")
        worldview = generate_worldview(user_prompt)

        print("=" * 60)
        print("【生成的世界观】")
        print("=" * 60)
        print(worldview)

        print("\n正在基于世界观生成角色设定，请稍等...\n")
        characters = generate_characters(user_prompt, worldview)

        print("=" * 60)
        print("【生成的角色设定】")
        print("=" * 60)
        print(characters)

    except Exception as e:
        print(f"\n发生错误：{e}")


if __name__ == "__main__":
    main()
    history.main()