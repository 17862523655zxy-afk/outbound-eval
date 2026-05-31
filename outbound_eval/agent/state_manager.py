"""State manager for agent internal state."""

from typing import Any, Optional


class StateManager:
    """Manages agent internal state."""

    def __init__(self):
        """Initialize the manager."""
        self.state: dict[str, Any] = {}
        self.state_history: list[dict] = []

    def update_state(self, key: str, value: Any) -> None:
        """Update a state value.

        Args:
            key: State key
            value: State value
        """
        old_value = self.state.get(key)
        self.state[key] = value

        # Record in history
        self.state_history.append(
            {
                "key": key,
                "old_value": old_value,
                "new_value": value,
            }
        )

    def get_state(self, key: Optional[str] = None) -> Any:
        """Get state value.

        Args:
            key: State key. If None, return all state.

        Returns:
            State value or full state dict
        """
        if key is None:
            return self.state.copy()
        return self.state.get(key)

    def has_state(self, key: str) -> bool:
        """Check if state key exists.

        Args:
            key: State key

        Returns:
            True if key exists
        """
        return key in self.state

    def reset(self) -> None:
        """Reset all state."""
        self.state = {}
        self.state_history = []

    def get_history(self) -> list[dict]:
        """Get state change history.

        Returns:
            List of state changes
        """
        return self.state_history.copy()

    def check_state_consistency(self, expected_state: dict) -> tuple[bool, list[str]]:
        """Check if current state matches expected state.

        Args:
            expected_state: Dict of expected key-value pairs

        Returns:
            Tuple of (is_consistent, list of inconsistencies)
        """
        inconsistencies = []

        for key, expected_value in expected_state.items():
            actual_value = self.state.get(key)

            if actual_value != expected_value:
                inconsistencies.append(
                    f"State '{key}': expected {expected_value}, got {actual_value}"
                )

        return len(inconsistencies) == 0, inconsistencies