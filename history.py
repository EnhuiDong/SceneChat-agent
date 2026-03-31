import os
import re
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings, load_index_from_storage
)
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownNodeParser
import chromadb

# ----------------------------
# 配置
# ----------------------------
load_dotenv()


persist_dir = "./storage_chroma"
data_dir = "./data"

# 推演参数
DEFAULT_BATCH_SIZE = 5
MAX_GLOBAL_HISTORY = 15

# 全局设置
def get_llm():
    provider = os.getenv("MODEL_PROVIDER", "dashscope").lower()
    model_name = os.getenv("LLM_MODEL")

    if provider == "dashscope":
        from llama_index.llms.dashscope import DashScope
        return DashScope(
            model=model_name,
            api_key=os.getenv("LLM_API_KEY")
        )

    elif provider == "openai":
      pass

    elif provider == "ollama":
        pass

    elif provider == "zhipuai":
        pass

    else:
        raise ValueError(f"other LLM provider: {provider}")


def get_embed_model():
    provider = os.getenv("EMBEDDING_PROVIDER", os.getenv("MODEL_PROVIDER", "dashscope")).lower()
    model_name = os.getenv("EMBEDDING_MODEL")
    api_key=os.getenv("EMBEDDING_API_KEY")
    if provider == "dashscope":
        from llama_index.embeddings.dashscope import DashScopeEmbedding
        return DashScopeEmbedding(model_name=model_name,api_key=api_key)

    elif provider == "openai":
        pass

    elif provider == "ollama":
        pass

    elif provider == "huggingface":
        pass

    else:
        raise ValueError(f"other Embedding provider: {provider}")
Settings.llm=get_llm()
Settings.embed_model=get_embed_model()

# ----------------------------
# 复用你已成功的 get_index 函数
# ----------------------------
def get_index():
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "sim_agent_rag"

    # if collection_name in [c.name for c in chroma_client.list_collections()]:
    #     print("📂 加载已有向量库...")
    #     chroma_collection = chroma_client.get_collection(collection_name)
    #     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    #     storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #     return load_index_from_storage(storage_context)
    docs = []
    for fn in ["world.md", "character.md"]:
        path = os.path.join(data_dir, fn)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                content = f.read()
            doc_type = "world" if "world" in fn else "character"
            from llama_index.core.schema import Document
            docs.append(Document(text=content, metadata={"file_name": fn, "doc_type": doc_type}))

    if not docs:
        raise FileNotFoundError("world.md or character.md is not found")

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(docs)

    for node in nodes:
        if node.metadata.get("doc_type") == "character":
            match = re.search(r'###\s*1\.\s*角色姓名\s*\n?([^\n#]+)', node.text, re.IGNORECASE)
            if match:
                node.metadata["character_name"] = match.group(1).strip()

    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index


# ----------------------------
# 数据结构
# ----------------------------
@dataclass
class Message:
    speaker: str
    action: str
    speech: str
    turn: int


class SimulationState:
    def __init__(self, scene: str):
        self.scene = scene
        self.history: List[Message] = []
        self.turn_count = 0

    def add_message(self, msg: Message):
        self.history.append(msg)
        self.turn_count += 1

    def get_recent_history(self, k: int = MAX_GLOBAL_HISTORY) -> str:
        recent = self.history[-k:]
        lines = [f"[{msg.turn}] {msg.speaker}: {msg.action} {msg.speech}" for msg in recent]
        return "\n".join(lines) if lines else "尚无对话。"


# ----------------------------
# 自主推演核心
# ----------------------------
def simulate_next_turn(state: SimulationState, rag_retriever) -> Optional[Message]:
    # RAG 检索上下文
    query = f"当前场景：{state.scene}。涉及的角色背景和世界观？"
    retrieved = rag_retriever.retrieve(query)
    rag_context = "\n".join([node.node.text for node in retrieved])

    history_str = state.get_recent_history(MAX_GLOBAL_HISTORY)

    prompt = f"""你正在主持一个社会模拟实验。

【世界观与角色设定】
{rag_context}

【当前场景】
{state.scene}

【已有对话历史（最近 {MAX_GLOBAL_HISTORY} 轮）】
{history_str}

请推进剧情：
- 决定下一个发言的角色（必须是设定中存在的角色）
- 生成该角色的动作和对话
- 动作要有画面感，对话要符合人设
- 不要解释，不要元评论

严格按以下格式输出一行：
角色姓名: <动作描述> <对话内容>

例如：
艾拉拉·范斯: 整理白大褂，语气平静地说：“病人情况稳定了。”
"""

    response = Settings.llm.complete(prompt, max_tokens=150)
    raw = response.text.strip()

    # 解析输出
    match = re.match(r'^(.*?):\s*(.*)$', raw)
    if not match:
        parts = raw.split(":", 1)
        if len(parts) != 2:
            return None
        speaker, rest = parts[0].strip(), parts[1].strip()
    else:
        speaker, rest = match.groups()
        speaker = speaker.strip()
        rest = rest.strip()

    if " " in rest:
        action, speech = rest.split(" ", 1)
    else:
        action = "说道"
        speech = rest

    speech = re.sub(r'^[“”""]|[“”""]$', '', speech).strip()

    return Message(
        speaker=speaker,
        action=action,
        speech=speech,
        turn=state.turn_count + 1
    )



def main():
    # 获取初始场景
    initial_scene = input("请输入初始场景（如：深夜，星港医疗中心走廊）：").strip()
    if not initial_scene:
        initial_scene = "清晨，星港医疗中心走廊"

    # 加载索引（复用你的成功逻辑）
    index = get_index()
    rag_retriever = index.as_retriever(similarity_top_k=3)

    state = SimulationState(initial_scene)
    print(f"\n场景设定：{initial_scene}\n")

    while True:
        try:
            q_input = input(f"\n输入推演轮数（默认 {DEFAULT_BATCH_SIZE}，'quit' 退出）：").strip()
            if q_input.lower() == "quit":
                break
            q = int(q_input) if q_input else DEFAULT_BATCH_SIZE

            print(f"\n开始推演 {q} 轮...\n")
            for _ in range(q):
                msg = simulate_next_turn(state, rag_retriever)
                if msg is None:
                    print("LLM 输出解析失败，跳过本轮")
                    continue
                state.add_message(msg)
                print(f"[{msg.turn}] {msg.speaker}: {msg.action} {msg.speech}")

            print(f"\n本批完成，总轮数：{state.turn_count}")

        except ValueError:
            print("请输入有效数字")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    # 可选保存
    if input("\n保存日志到 simulation_log.txt? (y/n): ").lower() == "y":
        with open("simulation_log.txt", "w", encoding="utf-8") as f:
            f.write(f"场景: {state.scene}\n\n")
            for msg in state.history:
                f.write(f"[{msg.turn}] {msg.speaker}: {msg.action} {msg.speech}\n")
        print("已保存")

    print("推演结束！")


if __name__ == "__main__":
    main()