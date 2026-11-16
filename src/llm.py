"""OpenAI chat client có đếm token/chi phí. Lỗi API được ném ra, không có fallback giả."""
from __future__ import annotations

import threading
import time
from typing import Any

from openai import OpenAI

from . import config


class ChatModel:
    def __init__(self, model: str | None = None, reasoning_effort: str = 'none') -> None:
        self.client = OpenAI(api_key=config.require_api_key(), max_retries=3, timeout=120)
        self.model = model or config.AGENT_MODEL
        self.reasoning_effort = reasoning_effort
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.seconds = 0.0
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * config.AGENT_PRICE_IN_PER_1M
                + self.completion_tokens * config.AGENT_PRICE_OUT_PER_1M) / 1e6

    def usage(self) -> dict[str, Any]:
        return {'model': self.model, 'calls': self.calls, 'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens, 'cost_usd': round(self.cost_usd, 6),
                'api_seconds': round(self.seconds, 2)}

    def chat(self, messages: list[dict[str, Any]], **kw: Any):
        kw.setdefault('reasoning_effort', self.reasoning_effort)
        kw.setdefault('temperature', 0)
        kw = {k: v for k, v in kw.items() if v is not None}  # None = không gửi tham số đó
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kw)
        with self._lock:  # an toàn khi gọi song song (vision)
            self.seconds += time.perf_counter() - t0
            self.calls += 1
            self.prompt_tokens += resp.usage.prompt_tokens
            self.completion_tokens += resp.usage.completion_tokens
        return resp
