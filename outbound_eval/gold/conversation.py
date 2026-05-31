"""Gold conversation models."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class DialogTurn(BaseModel):
    """A single dialog turn."""

    turn_number: int = Field(description="Turn number")
    speaker: Literal["agent", "user"] = Field(description="Speaker role")
    content: str = Field(description="Message content")
    annotations: Optional[dict] = Field(
        default=None, description="Annotations for this turn"
    )


class GoldConversation(BaseModel):
    """A gold/reference conversation for evaluation."""

    conversation_id: str = Field(description="Unique conversation ID")
    task_id: str = Field(description="Associated task ID")
    persona_type: str = Field(
        default="cooperative", description="User persona type"
    )
    scenario: str = Field(description="Scenario description")
    quality_level: Literal["excellent", "good", "acceptable"] = Field(
        default="good", description="Quality level"
    )

    # Conversation turns
    turns: list[DialogTurn] = Field(
        default_factory=list, description="List of dialog turns"
    )

    # Annotations
    annotations: dict = Field(
        default_factory=dict, description="Conversation-level annotations"
    )

    # Metadata
    created_at: Optional[str] = Field(
        default=None, description="Creation timestamp"
    )
    source: Literal["expert", "real_call", "generated"] = Field(
        default="expert", description="Source of the conversation"
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for categorization"
    )

    def get_turn_count(self) -> int:
        """Get total turn count."""
        return len(self.turns)

    def get_agent_turns(self) -> list[DialogTurn]:
        """Get agent turns."""
        return [t for t in self.turns if t.speaker == "agent"]

    def get_user_turns(self) -> list[DialogTurn]:
        """Get user turns."""
        return [t for t in self.turns if t.speaker == "user"]

    def to_dialog_history(self) -> list[dict]:
        """Convert to dialog history format.

        Returns:
            List of dicts with role and content
        """
        return [
            {"role": turn.speaker, "content": turn.content}
            for turn in self.turns
        ]