from .question_1 import LLMConfig, LLMStreamClient, StreamToken

class LLMFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def create_client(self, provider: str, **kwargs) -> LLMStreamClient:
        """
        Factory shortcut.

        provider: "openai" | "anthropic" | "ollama" | "gemini" | any base_url string
        """
        if provider == "openai":
            from .openai_stream import OpenAIStream
            cls = OpenAIStream
        elif provider == "anthropic":
            from .anthropic_stream import AnthropicStream
            cls = AnthropicStream
        elif provider == "ollama":
            from .ollam_stream import OllamaStream
            cls = OllamaStream
        elif provider == "gemini":
            from .gemini_stream import GeminiStream
            cls = GeminiStream
        else:
            # Assume provider is a base_url for a generic OpenAI-compatible client
            from .openai_stream import OpenAIStream
            cls = OpenAIStream
            kwargs['base_url'] = provider  # pass as base_url
        
        return cls(model=self.model_name, **kwargs)