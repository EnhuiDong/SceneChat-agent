import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.node_parser import MarkdownNodeParser
import re


load_dotenv()
api_key = os.getenv("LLM_API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in your environment")

persist_dir = "./storage_chroma"
data_dir = "./data"


Settings.embed_model = DashScopeEmbedding(model_name=os.getenv("EMBEDDING_MODEL"))
Settings.llm = DashScope(model=os.getenv("LLM_MODEL"), api_key=api_key)



def get_index():
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    if "sim_agent_rag" in [c.name for c in chroma_client.list_collections()]:
        chroma_collection = chroma_client.get_collection("sim_agent_rag")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return load_index_from_storage(storage_context)

    # 加载文档
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
        raise FileNotFoundError("world.md  or character.md is not found.")

    # 按 Markdown 结构切片
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(docs)

    # 增强角色元数据
    for node in nodes:
        if node.metadata.get("doc_type") == "character":
            match = re.search(r'###\s*1\.\s*角色姓名\s*\n?([^\n#]+)', node.text, re.IGNORECASE)
            if match:
                node.metadata["character_name"] = match.group(1).strip()

    # 存入 Chroma
    chroma_collection = chroma_client.get_or_create_collection("sim_agent_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index


# ----------------------------
# 4. 主程序
# ----------------------------
def main():
    index = get_index()
    query_engine = index.as_query_engine(
        streaming=False,
        similarity_top_k=10
    )

    while True:
        question = input("\n 问题: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        response = query_engine.query(question)
        print(f"\n{response}")


if __name__ == "__main__":
    main()