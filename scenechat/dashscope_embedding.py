"""Native DashScope text embeddings that preserve service failures."""

from llama_index.embeddings.dashscope import DashScopeEmbedding

from .errors import classify_provider_error
from .build_control import CURRENT_BUILD, await_controlled
from .config import config_int
from .recovery import transport_call
import asyncio


class CheckedDashScopeEmbedding(DashScopeEmbedding):
    def _checked_embeddings(self, texts: list[str], text_type: str) -> list[list[float]]:
        control = CURRENT_BUILD.get()
        if control is None:
            return self._embedding_request(texts, text_type)
        return transport_call(lambda: self._embedding_request(texts, text_type), control=control,
                              retries=0 if "preflight" in control.stage else config_int(
                                  "embedding", "max_retries", 1, minimum=0, maximum=2))

    def _embedding_request(self, texts: list[str], text_type: str) -> list[list[float]]:
        import dashscope

        control = CURRENT_BUILD.get()
        timeout = config_int("embedding", "request_timeout_seconds", 180, minimum=1, maximum=600)
        if control is None:
            response = dashscope.TextEmbedding.call(
                model=self.model_name, input=texts, api_key=self._api_key,
                text_type=text_type, request_timeout=timeout,
            )
        else:
            from dashscope.client.base_api import BaseAioApi
            from dashscope.common.utils import _get_task_group_and_task
            task_group, function = _get_task_group_and_task("dashscope.embeddings.text_embedding")

            async def request():
                return await await_controlled(BaseAioApi.call(
                    model=self.model_name, input={"texts": texts}, api_key=self._api_key,
                    task_group=task_group, task="text-embedding", function=function,
                    text_type=text_type, request_timeout=timeout,
                ), control, timeout)

            response = asyncio.run(request())
        if response.status_code != 200:
            # The upstream adapter logs this and returns [None], losing the
            # quota/authentication failure to a later Pydantic ValidationError.
            cause = RuntimeError(
                f"{response.status_code} {response.code}: {response.message}"
            )
            cause.status_code = response.status_code
            raise classify_provider_error(cause, service="向量模型") from cause
        output = response.output or {}
        rows = sorted(output.get("embeddings") or [], key=lambda item: item["text_index"])
        if len(rows) != len(texts) or [item["text_index"] for item in rows] != list(range(len(texts))):
            raise ValueError("向量模型返回的 embedding 数量或索引无效")
        if any(not isinstance(item.get("embedding"), list) or not item["embedding"] for item in rows):
            raise ValueError("向量模型返回空或非法 embedding")
        return [[float(value) for value in item["embedding"]] for item in rows]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._checked_embeddings(texts, self._text_type)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._checked_embeddings([text], self._text_type)[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._checked_embeddings([query], "query")[0]
