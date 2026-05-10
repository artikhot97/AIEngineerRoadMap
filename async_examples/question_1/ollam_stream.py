# ─────────────────────────────────────────────
#  Ollama (local)
# ─────────────────────────────────────────────

from .question_1 import LLMConfig, LLMStreamClient, StreamToken
import json
from typing import Optional

class OllamaStream(LLMStreamClient):
    """Ollama local inference — streams via /api/chat."""

    DEFAULT_BASE = "http://localhost:11434"

    def __init__(
        self,
        model: str = "llama3",
        base_url: Optional[str] = None,
        **kw,
    ) -> None:
        cfg = LLMConfig(
            base_url=base_url or self.DEFAULT_BASE,
            model=model,
            **kw,
        )
        super().__init__(cfg)

    def _build_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        h.update(self.config.extra_headers)
        return h

    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/api/chat"

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
        if not line:
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        text = obj.get("message", {}).get("content", "")
        done = obj.get("done", False)
        finish = "stop" if done else None
        return StreamToken(text=text, raw=obj, finish_reason=finish)