"""OpenAI client implementation."""

import os
from typing import Optional
from openai import OpenAI
from outbound_eval.agent.llm.base import LLMClient
from outbound_eval.infra.config import settings


class OpenAIClient(LLMClient):
    """OpenAI API client (supports DeepSeek via base_url)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the client.

        Args:
            api_key: API key. Defaults to env var.
            model: Model name. Defaults to settings.llm_model.
            base_url: API base URL (e.g., https://api.deepseek.com for DeepSeek)
        """
        self.api_key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.base_url or os.getenv("BASE_URL")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # Track token usage
        self.total_tokens = 0
        self.total_cost = 0.0

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion."""
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            **kwargs,
        )
        return response.choices[0].text

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate a chat completion."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )

            # Track usage
            if hasattr(response, "usage") and response.usage:
                self.total_tokens += response.usage.total_tokens
                self.total_cost += (response.usage.total_tokens / 1000) * settings.token_price_per_1k

            return response.choices[0].message.content
        except Exception as e:
            raise ConnectionError(f"Failed to call LLM API: {str(e)}") from e

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken approximation."""
        # Simple approximation: ~4 chars per token for English
        # and ~2 chars per token for Chinese
        chinese_chars = sum(1 for c in text if ord(c) > 127)
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 0.5 + other_chars * 0.25)

    def get_usage_stats(self) -> dict:
        """Get token usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }

    def reset_usage(self) -> None:
        """Reset usage counters."""
        self.total_tokens = 0
        self.total_cost = 0.0