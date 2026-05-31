"""State tracking evaluation."""

from typing import Optional, Any
from pydantic import BaseModel, Field


class StateConsistencyResult(BaseModel):
    """Result of state consistency check."""

    state_key: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    is_consistent: bool = True
    reason: Optional[str] = None


class StateTrackingResult(BaseModel):
    """Result of state tracking evaluation."""

    total_states: int
    consistent_states: int
    inconsistent_states: int
    consistency_score: float = Field(description="0-100 score")
    state_details: list[StateConsistencyResult] = Field(default_factory=list)
    critical_state_errors: list[str] = Field(default_factory=list)


class StateTrackingJudge:
    """Evaluates state tracking accuracy."""

    def __init__(self):
        """Initialize the judge."""
        self.agent_state: dict = {}
        self.ground_truth_state: dict = {}

    def set_agent_state(self, state: dict) -> None:
        """Set agent's internal state.

        Args:
            state: Agent's current state
        """
        self.agent_state = state

    def set_ground_truth(
        self, dialogue_history: list[dict], expected_outcome: dict
    ) -> None:
        """Set ground truth state from dialogue.

        Args:
            dialogue_history: Conversation history
            expected_outcome: Expected task outcome
        """
        self.ground_truth_state = {}

        user_messages = [
            t["content"] for t in dialogue_history if t["role"] == "user"
        ]

        # Infer user intent from messages
        all_text = " ".join(user_messages)

        # Identity confirmation
        if any(kw in all_text for kw in ["是", "对", "没错"]):
            self.ground_truth_state["identity_confirmed"] = True

        # Commitment
        if any(kw in all_text for kw in ["可以", "好的", "没问题", "愿意"]):
            self.ground_truth_state["commitment_obtained"] = True
            self.ground_truth_state["user_intent"] = "confirm"
        elif any(kw in all_text for kw in ["不送了", "不做了", "退出", "不想"]):
            self.ground_truth_state["commitment_obtained"] = False
            self.ground_truth_state["user_intent"] = "reject"

        # Safety reminder
        agent_messages = [
            t["content"] for t in dialogue_history if t["role"] == "agent"
        ]
        if any("安全" in msg for msg in agent_messages):
            self.ground_truth_state["safety_reminded"] = True

        # Update from expected outcome
        if expected_outcome:
            self.ground_truth_state.update(expected_outcome)

    def evaluate(self) -> StateTrackingResult:
        """Evaluate state tracking consistency.

        Returns:
            State tracking result
        """
        state_details: list[StateConsistencyResult] = []
        consistent = 0
        inconsistent = 0
        critical_errors: list[str] = []

        for key, gt_value in self.ground_truth_state.items():
            agent_value = self.agent_state.get(key)

            result = StateConsistencyResult(
                state_key=key,
                expected_value=gt_value,
                actual_value=agent_value,
            )

            # Check consistency
            if key in ["user_intent"]:
                # These should match exactly
                if agent_value != gt_value:
                    result.is_consistent = False
                    result.reason = f"Intent mismatch: expected {gt_value}, got {agent_value}"
                    inconsistent += 1
                    critical_errors.append(key)
                else:
                    consistent += 1
            else:
                # For boolean states, check if agent correctly recorded
                if isinstance(gt_value, bool):
                    if agent_value != gt_value:
                        result.is_consistent = False
                        result.reason = f"State mismatch for {key}"
                        inconsistent += 1
                        if key in ["commitment_obtained", "identity_confirmed"]:
                            critical_errors.append(key)
                    else:
                        consistent += 1

            state_details.append(result)

        total = len(self.ground_truth_state)
        score = (consistent / total * 100) if total > 0 else 100.0

        return StateTrackingResult(
            total_states=total,
            consistent_states=consistent,
            inconsistent_states=inconsistent,
            consistency_score=score,
            state_details=state_details,
            critical_state_errors=critical_errors,
        )