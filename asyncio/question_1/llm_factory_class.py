from question_1 import LLMConfig, LLMStreamClient, StreamToken
from openai_stream import OpenAIStream
from anthropic_stream import AnthropicStream
from ollam_stream import OllamaStream

class LLMFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    mapping = {
            "openai": OpenAIStream,
            "anthropic": AnthropicStream,
            "ollama": OllamaStream,
        }
        
    def create_client(self, provider: str, **kwargs) -> LLMStreamClient:
        """
        Factory shortcut.

        provider: "openai" | "anthropic" | "ollama" | any base_url string
        """
        cls = LLMFactory.mapping.get(provider)
        if cls is not None:
            return cls(model=self.model_name, **kwargs)
        else:
            # Assume provider is a base_url for a generic OpenAI-compatible client
            raise ValueError(f"Unknown provider '{provider}'. Valid options are: {list(LLMFactory.mapping.keys())}")