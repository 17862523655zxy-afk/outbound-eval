"""LLM client base."""

from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion.

        Args:
            prompt: The prompt to complete
            **kwargs: Additional parameters

        Returns:
            The generated completion text
        """
        pass

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            The generated response text
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        pass