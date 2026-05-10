# ─────────────────────────────────────────────
#  Anthropic (Claude)
# ─────────────────────────────────────────────
from .question_1 import LLMStreamClient, LLMConfig, StreamToken
from .llm_factory_class import LLMFactory
from typing import Optional, Any
import os
import json

class AnthropicStream(LLMStreamClient):
    """Native Anthropic Messages streaming (claude-* models)."""

    DEFAULT_BASE = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-haiku-20241022",
        max_tokens: int = 1024,
        **kw,
    ) -> None:
        cfg = LLMConfig(
            base_url=self.DEFAULT_BASE,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            model=model,
            extra_body={"max_tokens": max_tokens},
            **kw,
        )
        super().__init__(cfg)

    def _build_headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": self.API_VERSION,
        }
        h.update(self.config.extra_headers)
        return h

    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/v1/messages"

    def _build_payload(self, messages: list[dict], **kwargs) -> dict:
        # Separate system prompt from messages if present
        system = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": filtered,
            "stream": True,
            **self.config.extra_body,
            **kwargs,
        }
        if system:
            payload["system"] = system
        return payload

    def _parse_chunk(self, raw_line: str) -> Optional[StreamToken]:
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            return None
        data = line[len("data:"):].strip()
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None

        event_type = obj.get("type", "")

        if event_type == "content_block_delta":
            delta = obj.get("delta", {})
            text = delta.get("text", "")
            return StreamToken(text=text, raw=obj)

        if event_type == "message_stop":
            return StreamToken(text="", raw=obj, finish_reason="stop")

        if event_type == "message_delta":
            reason = obj.get("delta", {}).get("stop_reason")
            if reason:
                return StreamToken(text="", raw=obj, finish_reason=reason)

        return None  # ignore other event types (ping, content_block_start, …)

