"""Модель для ответов: DashScope в OpenAI-совместимом режиме, ответ приходит потоком.

Клиент — обычный HTTP-запрос: видно ровно то, что уходит модели и что приходит
обратно. qwen3.8 перед ответом всегда думает: ход мысли идёт отдельным полем
reasoning_content, сам ответ — полем content, расход токенов — последним куском.
"""
import json
from collections.abc import Iterator

import httpx

from . import config


class LLMError(RuntimeError):
    """Ошибка вызова модели — уже человеческими словами. transient — есть смысл повторить."""

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def _detail(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["message"]
    except Exception:
        return response.text[:300]


def stream(messages: list[dict], budget: int | None = None, as_json: bool = False) -> Iterator[tuple[str, object]]:
    """Ответ модели по кусочкам: ("think", текст), ("text", текст), в конце ("usage", {...}).
    budget — сколько токенов модели можно думать; выключить размышления у qwen3.8 нельзя.
    as_json — ответ строго JSON-объектом (какие в нём поля, говорит промпт)."""
    if not config.DASHSCOPE_API_KEY:
        raise LLMError("нет ключа модели: впишите DASHSCOPE_API_KEY в .env")
    body = {"model": config.LLM_MODEL, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}}
    if budget is not None:
        body["thinking_budget"] = budget
    if as_json:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {config.DASHSCOPE_API_KEY}"}
    try:
        with httpx.stream("POST", f"{config.DASHSCOPE_BASE_URL}/chat/completions", json=body,
                          headers=headers, timeout=httpx.Timeout(90, connect=15)) as response:
            if response.status_code != 200:
                response.read()
                raise LLMError(f"модель ответила {response.status_code}: {_detail(response)}",
                               transient=response.status_code == 429 or response.status_code >= 500)
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise LLMError(f"модель прервала ответ: {chunk['error'].get('message', chunk['error'])}", True)
                if chunk.get("usage"):
                    yield "usage", chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        yield "think", delta["reasoning_content"]
                    if delta.get("content"):
                        yield "text", delta["content"]
    except httpx.HTTPError as e:
        raise LLMError(f"модель не отвечает: {e.__class__.__name__} {e}".strip(), transient=True) from e
