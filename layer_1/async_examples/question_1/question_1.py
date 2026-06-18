#Build an async streaming wrapper for any LLM API that yields tokens via async def stream() generator
"""
Async streaming wrapper for any LLM API using httpx.
Yields tokens via `async def stream()` generator.

Supports: OpenAI, Anthropic, Ollama, and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional
import asyncio

import httpx

# Allow relative imports when run directly
import sys
from pathlib import Path
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'async_examples.question_1'


# ─────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────

@dataclass
class StreamToken:
    text: str # the text content of this token (may be empty for non-text events)
    raw: dict[str, Any] = field(default_factory=dict) # original raw data from the chunk, for advanced use cases
    finish_reason: Optional[str] = None # e.g. "stop", "length", "error", etc., if this token indicates the end of the stream

    @property
    def is_done(self) -> bool:
        return self.finish_reason is not None


@dataclass
class LLMConfig:
    base_url: str
    api_key: Optional[str] = None
    model: str = ""
    timeout: float = 60.0
    extra_headers: dict[str, str] = field(default_factory=dict) # any additional headers to include in requests
    extra_body: dict[str, Any] = field(default_factory=dict) # any additional fields to include in the request body (e.g. temperature, top_p, etc.)


# ─────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────

class LLMStreamClient(ABC):
    """Abstract base: subclass for each provider."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._headers = self._build_headers()

    @abstractmethod
    def _build_headers(self) -> dict[str, str]:
        """Construct request headers, including auth."""
        ...

    @abstractmethod
    def _build_payload(self, messages: list[dict], **kwargs) -> dict:
        """Construct the JSON body for the request, including messages and any extra params."""
        ...

    @abstractmethod
    def _endpoint(self) -> str:
        """Return the full URL for the streaming endpoint."""
        ...

    @abstractmethod
    def _parse_chunk(self, raw_line: str) -> Optional[StreamToken]:
        """Return a StreamToken, or None to skip this line."""
        ...

    async def stream(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AsyncGenerator[StreamToken, None]:
        """
        Async generator that yields StreamToken objects.

        Usage:
            async for token in client.stream([{"role": "user", "content": "Hi"}]):
                print(token.text, end="", flush=True)
        """
        payload = self._build_payload(messages, **kwargs)
        url = self._endpoint()

        async with httpx.AsyncClient(timeout=self.config.timeout) as http:
            async with http.stream("POST", url, headers=self._headers, json=payload) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    token = self._parse_chunk(raw_line)
                    if token is not None:
                        yield token
                        if token.is_done:
                            return

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Convenience: collect full streamed response as a string."""
        parts: list[str] = []
        async for token in self.stream(messages, **kwargs):
            parts.append(token.text)
        return "".join(parts)







# ─────────────────────────────────────────────
#  Demo / quick test
# ─────────────────────────────────────────────


async def _demo() -> None:
    from .llm_factory_class import LLMFactory

    messages = [{"role": "user", "content": "Count from 1 to 5, one number per line."}]

    # Swap in any client:
    factory = LLMFactory(model_name="gemini-3-flash-preview")
    client = factory.create_client(provider="gemini")

    print("── streaming ──")
    async for token in client.stream(messages):
        print(token.text, end="", flush=True)
    print("\n── done ──")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo())