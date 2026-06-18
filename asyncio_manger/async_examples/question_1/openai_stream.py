# ─────────────────────────────────────────────
#  OpenAI (and compatible: Together, Groq, etc.)
# ─────────────────────────────────────────────

from .question_1 import LLMStreamClient, LLMConfig, StreamToken
from .llm_factory_class import LLMFactory
from typing import Optional
import os
import json


class OpenAIStream(LLMStreamClient):
    """
    Works with OpenAI, Together AI, Groq, Fireworks, Anyscale,
    and any OpenAI-compatible /v1/chat/completions endpoint.
    """

    DEFAULT_BASE = "https://api.openai.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        **kw,
    ) -> None:
        cfg = LLMConfig(
            base_url=base_url or self.DEFAULT_BASE,
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            model=model,
            **kw,
        )
        super().__init__(cfg)

    def _build_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"}
        h.update(self.config.extra_headers)
        return h

    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/v1/chat/completions"

    def _build_payload(self, messages: list[dict], **kwargs) -> dict:
        return {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            **self.config.extra_body,
            **kwargs,
        }

    def _parse_chunk(self, raw_line: str) -> Optional[StreamToken]:
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            return None
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return StreamToken(text="", finish_reason="stop")
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None
        choice = obj.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        text = delta.get("content") or ""
        finish = choice.get("finish_reason")
        return StreamToken(text=text, raw=obj, finish_reason=finish)

