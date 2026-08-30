import re
from dataclasses import dataclass
from typing import Any, List

from .character_parser import split_character_profiles
from .providers import get_embedding_model
from .visibility import ViewerContext, access_keys_for, normalize_scopes


EMBEDDING_INSERT_BATCH_SIZE = 20
RAG_MIN_DOCUMENT_CHARS = 800


def _character_name(profile: str) -> str:
    match = re.search(
        r"^###\s*1\.\s*角色姓名\s*\n+([^\n#]+)", profile, re.MULTILINE
    )
    return match.group(1).strip().strip("*") if match else ""


def build_knowledge_documents(
    worldview: str,
    characters: str,
    experiment_id: str,
    director_notes: str = "",
    facts: list[Any] | None = None,
):
    """Build documents with metadata that enforces experiment and role boundaries."""
    from llama_index.core.schema import Document

    documents = [
        Document(
            text=worldview,
            metadata={
                "experiment_id": experiment_id,
                "doc_type": "world",
                "visibility": "public",
                "access_key": "public",
            },
        )
    ]
    if director_notes.strip():
        documents.append(
            Document(
                text=director_notes,
                metadata={
                    "experiment_id": experiment_id,
                    "doc_type": "director",
                    "visibility": "director_only",
                    "access_key": "director",
                },
            )
        )
    for fact in facts or []:
        content = str(getattr(fact, "content", "") or "").strip()
        if not content:
            continue
        fact_id = str(getattr(fact, "id", "") or "fact")
        for access_key in normalize_scopes(getattr(fact, "visibility", None)):
            key = "director" if access_key == "director_only" else access_key
            documents.append(
                Document(
                    text=content,
                    metadata={
                        "experiment_id": experiment_id,
                        "doc_type": "fact",
                        "fact_id": fact_id,
                        "visibility": access_key,
                        "access_key": key,
                    },
                )
            )
    for profile in split_character_profiles(characters):
        name = _character_name(profile)
        if name:
            documents.append(
                Document(
                    text=profile,
                    metadata={
                        "experiment_id": experiment_id,
                        "doc_type": "character",
                        "character_name": name,
                        "visibility": "agent_private",
                        "access_key": f"agent:{name}",
                    },
                )
            )
    return documents


@dataclass
class ExperimentKnowledgeBase:
    """An experiment-local vector index with agent-aware retrieval filters."""

    experiment_id: str
    client: Any
    index: Any
    documents: list[Any] | None = None

    def _direct_context(self, access_keys: set[str] | None = None) -> str:
        texts = []
        for document in self.documents or []:
            if access_keys is not None and document.metadata.get("access_key") not in access_keys:
                continue
            if self.index is not None and len(document.text) >= RAG_MIN_DOCUMENT_CHARS:
                continue
            texts.append(document.text)
        return "\n\n".join(dict.fromkeys(texts))

    def retrieve_for_agent(
        self,
        agent_name: str,
        query: str,
        top_k: int = 4,
        *,
        role: str = "",
        location: str = "",
    ) -> str:
        from llama_index.core.vector_stores import (
            FilterCondition,
            MetadataFilter,
            MetadataFilters,
        )

        viewer = ViewerContext(name=agent_name, role=role, location=location)
        keys = access_keys_for(viewer)
        if self.index is None:
            return self._direct_context(set(keys))
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="access_key", value=key)
                for key in keys
            ],
            condition=FilterCondition.OR,
        )
        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            filters=filters,
        )
        nodes = retriever.retrieve(query)
        vector_text = "\n\n".join(dict.fromkeys(node.node.text for node in nodes))
        direct_text = self._direct_context(set(keys))
        return "\n\n".join(part for part in (direct_text, vector_text) if part)

    def retrieve_for_narrator(
        self,
        query: str,
        top_k: int = 6,
        include_private: bool = False,
    ) -> str:
        """Retrieve public world context or the full corpus for reader-only narration."""
        access_keys = None if include_private else {"public"}
        if self.index is None:
            return self._direct_context(access_keys)
        kwargs = {"similarity_top_k": top_k}
        if not include_private:
            from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

            kwargs["filters"] = MetadataFilters(
                filters=[MetadataFilter(key="access_key", value="public")]
            )
        retriever = self.index.as_retriever(**kwargs)
        nodes = retriever.retrieve(query)
        vector_text = "\n\n".join(dict.fromkeys(node.node.text for node in nodes))
        direct_text = self._direct_context(access_keys)
        return "\n\n".join(part for part in (direct_text, vector_text) if part)


def build_experiment_knowledge_base(
    worldview: str,
    characters: str,
    experiment_id: str,
    embed_model=None,
    director_notes: str = "",
    facts: list[Any] | None = None,
) -> ExperimentKnowledgeBase:
    """Create an in-memory Chroma collection unique to one experiment."""
    import chromadb
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.vector_stores.chroma import ChromaVectorStore

    documents = build_knowledge_documents(
        worldview,
        characters,
        experiment_id,
        director_notes=director_notes,
        facts=facts,
    )
    rag_documents = [
        document for document in documents if len(document.text) >= RAG_MIN_DOCUMENT_CHARS
    ]
    if not rag_documents:
        return ExperimentKnowledgeBase(experiment_id, None, None, documents)

    nodes = MarkdownNodeParser().get_nodes_from_documents(rag_documents)

    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"scenechat-{experiment_id}")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model or get_embedding_model(),
        insert_batch_size=EMBEDDING_INSERT_BATCH_SIZE,
    )
    return ExperimentKnowledgeBase(experiment_id, client, index, documents)


def requires_vector_index(
    worldview: str,
    characters: str,
    director_notes: str = "",
    facts: list[Any] | None = None,
) -> bool:
    texts = [worldview, director_notes, *split_character_profiles(characters)]
    texts.extend(str(getattr(fact, "content", "") or "") for fact in facts or [])
    return any(len(text) >= RAG_MIN_DOCUMENT_CHARS for text in texts)
