from __future__ import annotations

from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import PrivateAttr


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Expose an OpenAI-compatible /embeddings endpoint to LlamaIndex."""

    _client: Any = PrivateAttr()

    def __init__(
        self,
        *,
        client: Any,
        model_name: str,
        embed_batch_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "OpenAICompatibleEmbedding"

    @staticmethod
    def _item_value(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    def _embed(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
            encoding_format="float",
        )
        data = list(getattr(response, "data", None) or [])
        ordered = sorted(
            enumerate(data),
            key=lambda pair: self._item_value(pair[1], "index", pair[0]),
        )
        embeddings = [
            [float(value) for value in self._item_value(item, "embedding", [])]
            for _, item in ordered
        ]
        if len(embeddings) != len(texts) or any(not vector for vector in embeddings):
            raise ValueError("向量模型返回的 embedding 数量或内容无效")
        return embeddings

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embed(texts)
