"""Dialog manager for conversation history."""

from typing import Optional


class DialogManager:
    """Manages conversation dialog history."""

    def __init__(self, max_history: int = 50):
        """Initialize the manager.

        Args:
            max_history: Maximum number of turns to keep
        """
        self.max_history = max_history
        self.history: list[dict[str, str]] = []
        self.turn_count = 0

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn.

        Args:
            role: 'agent' or 'user'
            content: Message content
        """
        self.turn_count += 1
        self.history.append(
            {
                "role": role,
                "content": content,
                "turn": self.turn_count,
            }
        )

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_history(self, last_n: Optional[int] = None) -> list[dict[str, str]]:
        """Get conversation history.

        Args:
            last_n: Return only last N turns. If None, return all.

        Returns:
            List of conversation turns
        """
        if last_n is None:
            return self.history.copy()
        return self.history[-last_n:]

    def get_last_turn(self, role: Optional[str] = None) -> Optional[dict]:
        """Get the last turn.

        Args:
            role: Optional filter by role

        Returns:
            Last turn dict or None
        """
        for turn in reversed(self.history):
            if role is None or turn["role"] == role:
                return turn
        return None

    def reset(self) -> None:
        """Reset the dialog history."""
        self.history = []
        self.turn_count = 0

    def get_turn_count(self) -> int:
        """Get the total number of turns.

        Returns:
            Turn count
        """
        return self.turn_count

    def get_messages_by_role(self, role: str) -> list[str]:
        """Get all messages by a specific role.

        Args:
            role: 'agent' or 'user'

        Returns:
            List of message contents
        """
        return [turn["content"] for turn in self.history if turn["role"] == role]