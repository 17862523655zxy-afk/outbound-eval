"""Constraint engine for skill scripts."""

from typing import Optional


class ConstraintEngine:
    """Engine for evaluating and enforcing constraints."""

    def __init__(self, constraints: dict):
        """Initialize the engine.

        Args:
            constraints: Constraint configuration dict
        """
        self.constraints = constraints
        self.max_response_length = constraints.get("max_response_length", 30)
        self.style = constraints.get("style", "casual")
        self.avoid_repetition = constraints.get("avoid_repetition", True)
        self.interrupt_handling = constraints.get("interrupt_handling", "polite_transition")

        # Prohibited words
        self.prohibited_words = constraints.get("prohibited_words", [])

    def check_response_length(self, response: str) -> tuple[bool, Optional[str]]:
        """Check if response length is within limits.

        Args:
            response: The response text

        Returns:
            Tuple of (is_valid, error_message)
        """
        char_count = len(response)
        word_count = len(response.split())

        # Check character count
        if char_count > self.max_response_length:
            return False, f"Response too long: {char_count} chars (max {self.max_response_length})"

        return True, None

    def check_prohibited_words(self, response: str) -> tuple[bool, list[str]]:
        """Check for prohibited words.

        Args:
            response: The response text

        Returns:
            Tuple of (is_valid, list of found prohibited words)
        """
        found = []
        response_lower = response.lower()

        for word in self.prohibited_words:
            if word.lower() in response_lower:
                found.append(word)

        return len(found) == 0, found

    def check_all(self, response: str) -> tuple[bool, list[str]]:
        """Check all constraints.

        Args:
            response: The response text

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check length
        is_valid, error = self.check_response_length(response)
        if not is_valid:
            errors.append(error)

        # Check prohibited words
        is_valid, words = self.check_prohibited_words(response)
        if not is_valid:
            errors.append(f"Prohibited words found: {', '.join(words)}")

        return len(errors) == 0, errors

    def truncate_response(self, response: str, max_length: Optional[int] = None) -> str:
        """Truncate response to max length.

        Args:
            response: The response text
            max_length: Max length (uses default if not specified)

        Returns:
            Truncated response
        """
        limit = max_length or self.max_response_length
        if len(response) <= limit:
            return response

        # Try to truncate at sentence boundary
        truncated = response[:limit]
        last_punct = max(
            truncated.rfind("。"),
            truncated.rfind("，"),
            truncated.rfind("、"),
            truncated.rfind("！"),
            truncated.rfind("？"),
        )

        if last_punct > limit * 0.7:  # At least 70% of limit
            return truncated[: last_punct + 1]

        return truncated.rstrip(".,!?，、。！？")


# Default constraints for casual phone conversation
DEFAULT_CONSTRAINTS = {
    "max_response_length": 30,
    "style": "casual",
    "avoid_repetition": True,
    "interrupt_handling": "polite_transition",
    "prohibited_words": ["好的", "哈哈", "嘿嘿", "嘻嘻"],  # Avoid overly casual fillers
}