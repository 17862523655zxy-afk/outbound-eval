"""Flow adherence evaluation."""

from typing import Optional
from pydantic import BaseModel, Field


class FlowStepStatus(BaseModel):
    """Status of a flow step."""

    step_id: str
    step_name: str
    expected_state: str = "pending"
    actual_state: str = "pending"
    turn_number: Optional[int] = None
    deviation_reason: Optional[str] = None


class FlowAdherenceResult(BaseModel):
    """Result of flow adherence evaluation."""

    total_steps: int
    completed_steps: int
    skipped_steps: int
    deviated_steps: int
    adherence_score: float = Field(description="0-100 score")
    step_details: list[FlowStepStatus] = Field(default_factory=list)
    critical_path_compliant: bool = Field(
        default=True, description="Whether critical steps are complete"
    )


class FlowAdherenceJudge:
    """Evaluates flow adherence."""

    def __init__(self, flow_steps: list[dict], critical_step_ids: Optional[list[str]] = None):
        """Initialize the judge.

        Args:
            flow_steps: Flow step definitions
            critical_step_ids: IDs of critical steps that must be completed
        """
        self.flow_steps = flow_steps or []
        self.critical_step_ids = critical_step_ids or []

    def evaluate(
        self, dialogue_history: list[dict]
    ) -> FlowAdherenceResult:
        """Evaluate flow adherence.

        Args:
            dialogue_history: List of conversation turns

        Returns:
            Flow adherence result
        """
        step_statuses: list[FlowStepStatus] = []
        completed = 0
        skipped = 0
        deviated = 0

        for step in self.flow_steps:
            status = self._evaluate_step(step, dialogue_history)
            step_statuses.append(status)

            if status.actual_state == "completed":
                completed += 1
            elif status.actual_state == "skipped":
                skipped += 1
            elif status.actual_state == "deviated":
                deviated += 1

        # Calculate adherence score
        total = len(self.flow_steps)
        if total > 0:
            score = (completed * 1.0 + skipped * 0.5 + deviated * 0.0) / total * 100
        else:
            score = 100.0

        # Check critical path compliance
        critical_compliant = all(
            s.actual_state == "completed"
            for s in step_statuses
            if s.step_id in self.critical_step_ids
        )

        return FlowAdherenceResult(
            total_steps=total,
            completed_steps=completed,
            skipped_steps=skipped,
            deviated_steps=deviated,
            adherence_score=score,
            step_details=step_statuses,
            critical_path_compliant=critical_compliant,
        )

    def _evaluate_step(
        self, step: dict, dialogue_history: list[dict]
    ) -> FlowStepStatus:
        """Evaluate a single flow step.

        Args:
            step: Step definition
            dialogue_history: Conversation history

        Returns:
            Step status
        """
        status = FlowStepStatus(
            step_id=step.get("id", ""),
            step_name=step.get("description", ""),
        )

        expected_keywords = step.get("expected_keywords", {})

        # Handle both dict and list cases
        if isinstance(expected_keywords, dict):
            required = expected_keywords.get("required", [])
            confirm = expected_keywords.get("confirm", [])
            reject = expected_keywords.get("reject", [])
        else:
            # If it's a list, treat it as required keywords
            required = expected_keywords if isinstance(expected_keywords, list) else []
            confirm = []
            reject = []

        # Check agent messages for keywords
        agent_messages = [
            t["content"] for t in dialogue_history if t["role"] == "agent"
        ]
        user_messages = [
            t["content"] for t in dialogue_history if t["role"] == "user"
        ]

        # Check required keywords in agent messages
        if required:
            if any(kw in msg for msg in agent_messages for kw in required):
                status.actual_state = "completed"
            else:
                status.actual_state = "deviated"
                status.deviation_reason = f"Missing keywords: {required}"

        # Check user confirmation keywords
        if confirm:
            if any(kw in msg for msg in user_messages for kw in confirm):
                status.actual_state = "completed"

        # Check rejection keywords (if present, step is rejected)
        if reject:
            if any(kw in msg for msg in user_messages for kw in reject):
                # This might be a valid response if retention was attempted
                if "挽留" in str(agent_messages) or "鼓励" in str(agent_messages):
                    status.actual_state = "completed"
                else:
                    status.deviation_reason = "User rejected without retention"

        return status