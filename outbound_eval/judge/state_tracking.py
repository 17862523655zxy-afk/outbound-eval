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
        self.agent_state: dict[str, object] = {}
        self.ground_truth_state: dict[str, object] = {}

    def set_agent_state(self, state: dict[str, object]) -> None:
        """Set agent's internal state.

        Args:
            state: Agent's current state
        """
        self.agent_state = state

    def set_ground_truth(
        self, dialogue_history: list[dict[str, object]], expected_outcome: dict[str, object]
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

    def extract_state_path(self, dialogue_history: list[dict[str, object]]) -> list[str]:
        """Extract agent state path from conversation.

        States: INIT -> CONTRACT_NOTIFIED -> REQUIREMENT_EXPLAINED
                -> RETENTION_ATTEMPTED -> USER_CONFIRMED -> CALL_ENDED

        Args:
            dialogue_history: Conversation history

        Returns:
            Ordered list of state names reached.
        """
        states: list[str] = ["INIT"]
        agent_msgs = [t["content"] for t in dialogue_history if t["role"] == "agent"]
        all_agent_text = " ".join(agent_msgs)

        # CONTRACT_NOTIFIED: agent mentions contract / agreement
        if any(kw in all_agent_text for kw in ["合同", "协议", "签约", "签署", "已生效", "生效了"]):
            states.append("CONTRACT_NOTIFIED")

        # REQUIREMENT_EXPLAINED: agent mentions requirements / rules / delivery
        if any(kw in all_agent_text for kw in ["要求", "规定", "条款", "单数", "配送", "每天", "至少", "排名", "拒单"]):
            states.append("REQUIREMENT_EXPLAINED")

        # RETENTION_ATTEMPTED: agent attempts to retain / persuade
        if any(kw in all_agent_text for kw in ["挽留", "再考虑", "补贴", "奖励", "好处", "优势", "稳定", "放心", "解决"]):
            states.append("RETENTION_ATTEMPTED")

        # USER_CONFIRMED: user shows agreement
        user_msgs = [t["content"] for t in dialogue_history if t["role"] == "user"]
        all_user_text = " ".join(user_msgs)
        if any(kw in all_user_text for kw in ["可以", "好的", "没问题", "愿意", "开始", "配合", "行", "行吧"]):
            states.append("USER_CONFIRMED")

        # CALL_ENDED: agent says goodbye / thanks
        if agent_msgs:
            last_agent = agent_msgs[-1]
            if any(kw in last_agent for kw in ["再见", "拜拜", "感谢", "谢谢", "祝你", "注意安全"]):
                states.append("CALL_ENDED")

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_states: list[str] = []
        for s in states:
            if s not in seen:
                seen.add(s)
                unique_states.append(s)
        return unique_states

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