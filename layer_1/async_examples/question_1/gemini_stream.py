# ─────────────────────────────────────────────
#  Google Gemini (via AI Studio)
# ─────────────────────────────────────────────
from .question_1 import LLMStreamClient, LLMConfig, StreamToken
from typing import Optional, Any
import os
import json
from dotenv import load_dotenv
load_dotenv()  # Load .env file for API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class GeminiStream(LLMStreamClient):
    """Google Gemini streaming via AI Studio API."""

    DEFAULT_BASE = "https://generativelanguage.googleapis.com"
    API_VERSION = "v1beta"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        max_tokens: int = 1024,
        **kw,
    ) -> None:
        cfg = LLMConfig(
            base_url=self.DEFAULT_BASE,
            api_key=api_key or GEMINI_API_KEY,
            model=model,
            extra_body={"generationConfig": {"maxOutputTokens": max_tokens}},
            **kw,
        )
        super().__init__(cfg)

    def _build_headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
        }
        h.update(self.config.extra_headers)
        return h

    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/{self.API_VERSION}/models/{self.config.model}:streamGenerateContent?key={self.config.api_key}"

    def _build_payload(self, messages: list[dict], **kwargs) -> dict:
        # Convert OpenAI-style messages to Gemini format
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": m["content"]}]
            })
        
        payload = {
            "contents": contents,
            **self.config.extra_body,
        }
        return payload

    async def stream(self, messages: list[dict], **kwargs) -> AsyncGenerator[StreamToken, None]:
        import httpx

        payload = self._build_payload(messages, **kwargs)
        url = self._endpoint()
        headers = self._build_headers()

        async with httpx.AsyncClient() as http:
            async with http.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    # Try to parse complete JSON objects
                    start = 0
                    while start < len(buffer):
                        try:
                            obj, end = self._parse_json_object(buffer, start)
                            if obj:
                                # Process the object
                                if "candidates" in obj:
                                    for candidate in obj["candidates"]:
                                        if "content" in candidate and "parts" in candidate["content"]:
                                            for part in candidate["content"]["parts"]:
                                                if "text" in part:
                                                    yield StreamToken(text=part["text"], raw=obj)
                                        if candidate.get("finishReason"):
                                            yield StreamToken(text="", finish_reason=candidate["finishReason"].lower(), raw=obj)
                                start = end
                            else:
                                break
                        except:
                            break
                    buffer = buffer[start:]

    def _parse_json_object(self, buffer, start):
        # Find the start of the object
        obj_start = buffer.find('{', start)
        if obj_start == -1:
            return None, start
        # Then parse from there
        brace_count = 0
        in_string = False
        escape = False
        for i in range(obj_start, len(buffer)):
            c = buffer[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"' and not in_string:
                in_string = True
            elif c == '"' and in_string:
                in_string = False
            elif not in_string:
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            obj = json.loads(buffer[obj_start:i+1])
                            return obj, i+1
                        except:
                            return None, start
        return None, start

    def _parse_chunk(self, raw_line: str) -> Optional[StreamToken]:
        # Not used since we override stream
        return None