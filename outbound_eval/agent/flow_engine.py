"""Flow engine for conversation flow control."""

from typing import Optional


class FlowEngine:
    """Engine for managing conversation flow."""

    def __init__(self):
        """Initialize the engine."""
        self.flow_steps: list[dict] = []
        self.current_step_index = 0
        self.completed_steps: list[str] = []
        self.skipped_steps: list[str] = []
        self.step_records: list[dict] = []

    def reset(self, flow_steps: list[dict]) -> None:
        """Reset the engine with new flow steps.

        Args:
            flow_steps: List of flow step definitions
        """
        self.flow_steps = flow_steps or []
        self.current_step_index = 0
        self.completed_steps = []
        self.skipped_steps = []
        self.step_records = []

    def get_current_step(self) -> Optional[dict]:
        """Get the current step.

        Returns:
            Current step dict or None
        """
        if 0 <= self.current_step_index < len(self.flow_steps):
            return self.flow_steps[self.current_step_index]
        return None

    def advance_step(self) -> None:
        """Advance to the next step."""
        if self.current_step_index < len(self.flow_steps):
            step = self.flow_steps[self.current_step_index]
            self.completed_steps.append(step["id"])
            self.current_step_index += 1

    def skip_current_step(self, reason: str = "") -> None:
        """Skip the current step.

        Args:
            reason: Reason for skipping
        """
        if self.current_step_index < len(self.flow_steps):
            step = self.flow_steps[self.current_step_index]
            self.skipped_steps.append(step["id"])
            self.step_records.append(
                {
                    "step_id": step["id"],
                    "action": "skipped",
                    "reason": reason,
                }
            )
            self.current_step_index += 1

    def record_turn(self, user_message: str, agent_response: str) -> None:
        """Record a conversation turn.

        Args:
            user_message: User's message
            agent_response: Agent's response
        """
        current_step = self.get_current_step()
        if current_step:
            self.step_records.append(
                {
                    "step_id": current_step["id"],
                    "action": "turn_recorded",
                    "user_message": user_message,
                    "agent_response": agent_response,
                }
            )

    def get_progress(self) -> dict:
        """Get flow progress.

        Returns:
            Progress dict
        """
        total_steps = len(self.flow_steps)
        completed = len(self.completed_steps)
        skipped = len(self.skipped_steps)

        return {
            "total_steps": total_steps,
            "completed_steps": completed,
            "skipped_steps": skipped,
            "current_step_index": self.current_step_index,
            "current_step": self.get_current_step(),
            "progress_percentage": (
                (completed + skipped) / total_steps * 100 if total_steps > 0 else 0
            ),
        }

    def check_step_completion(
        self, step_id: str, user_message: str, agent_response: str
    ) -> bool:
        """Check if a step has been completed.

        Args:
            step_id: Step ID to check
            user_message: User's message
            agent_response: Agent's response

        Returns:
            True if step is complete
        """
        for step in self.flow_steps:
            if step["id"] == step_id:
                expected = step.get("expected_keywords", {})

                # Check agent keywords
                agent_kw = expected.get("required", [])
                if agent_kw:
                    if not any(kw in agent_response for kw in agent_kw):
                        return False

                # Check user keywords (for confirmation steps)
                confirm_kw = expected.get("confirm", [])
                if confirm_kw:
                    if not any(kw in user_message for kw in confirm_kw):
                        return False

                reject_kw = expected.get("reject", [])
                if reject_kw:
                    if any(kw in user_message for kw in reject_kw):
                        return False

                return True

        return False