"""Prompt builder utilities."""

from typing import Optional


class PromptBuilder:
    """Builder for agent prompts."""

    def __init__(self):
        """Initialize the builder."""
        self.system_prompt = ""
        self.context = {}

    def with_role(self, role: str) -> "PromptBuilder":
        """Set the role description.

        Args:
            role: Role description text

        Returns:
            Self for chaining
        """
        self.system_prompt = f"{role}\n\n"
        return self

    def with_constraints(self, constraints: list[str]) -> "PromptBuilder":
        """Add constraints.

        Args:
            constraints: List of constraint strings

        Returns:
            Self for chaining
        """
        self.system_prompt += "Constraints:\n"
        for c in constraints:
            self.system_prompt += f"- {c}\n"
        self.system_prompt += "\n"
        return self

    def with_flow(self, flow: str) -> "PromptBuilder":
        """Add conversation flow.

        Args:
            flow: Flow description

        Returns:
            Self for chaining
        """
        self.system_prompt += f"Conversation Flow:\n{flow}\n\n"
        return self

    def with_faq(self, faq: list[dict]) -> "PromptBuilder":
        """Add FAQ.

        Args:
            faq: List of Q&A dicts

        Returns:
            Self for chaining
        """
        self.system_prompt += "FAQ:\n"
        for item in faq:
            self.system_prompt += f"Q: {item.get('question', '')}\n"
            self.system_prompt += f"A: {item.get('answer', '')}\n\n"
        return self

    def with_context(self, **kwargs) -> "PromptBuilder":
        """Add context variables.

        Args:
            **kwargs: Context key-value pairs

        Returns:
            Self for chaining
        """
        self.context.update(kwargs)
        return self

    def build_system(self) -> str:
        """Build the system prompt.

        Returns:
            Complete system prompt
        """
        return self.system_prompt.strip()

    def build_user_message(
        self,
        user_message: str,
        dialogue_history: Optional[list[dict]] = None,
    ) -> str:
        """Build a user message with context.

        Args:
            user_message: Current user message
            dialogue_history: Optional conversation history

        Returns:
            Formatted user message
        """
        parts = []

        if self.context:
            parts.append("Context:")
            for k, v in self.context.items():
                parts.append(f"- {k}: {v}")
            parts.append("")

        if dialogue_history:
            parts.append("Conversation History:")
            for turn in dialogue_history[-5:]:  # Last 5 turns
                role = turn.get("role", "user")
                content = turn.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("")

        parts.append(f"User: {user_message}")

        return "\n".join(parts)

    def reset(self) -> "PromptBuilder":
        """Reset the builder.

        Returns:
            Self for chaining
        """
        self.system_prompt = ""
        self.context = {}
        return self