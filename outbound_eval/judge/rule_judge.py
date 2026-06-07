"""Rule-based evaluation."""

from typing import Optional
from pydantic import BaseModel, Field


class ConstraintViolation(BaseModel):
    """A constraint violation."""

    constraint_type: str
    message: str
    severity: str = "warning"  # error, warning, info


class RuleJudgeResult(BaseModel):
    """Result of rule-based evaluation."""

    is_compliant: bool = True
    violations: list[ConstraintViolation] = Field(default_factory=list)
    compliance_score: float = Field(description="0-100 score")


class RuleJudge:
    """Rule-based judge for fast compliance checks."""

    def __init__(self, constraints: dict[str, object]):
        """Initialize the judge.

        Args:
            constraints: Constraint configuration
        """
        self.constraints = constraints
        self.max_response_length = constraints.get("max_response_length", 30)
        self.prohibited_words = constraints.get("prohibited_words", [])

    def evaluate_message(self, message: str) -> RuleJudgeResult:
        """Evaluate a single message.

        Args:
            message: Message to evaluate

        Returns:
            Evaluation result
        """
        violations: list[ConstraintViolation] = []

        # Check length
        if len(message) > self.max_response_length:
            violations.append(
                ConstraintViolation(
                    constraint_type="response_length",
                    message=f"Response too long: {len(message)} chars (max {self.max_response_length})",
                    severity="error",
                )
            )

        # Check prohibited words
        for word in self.prohibited_words:
            if word in message:
                violations.append(
                    ConstraintViolation(
                        constraint_type="prohibited_words",
                        message=f"Prohibited word found: {word}",
                        severity="warning",
                    )
                )

        # Calculate score
        base_score = 100.0
        for v in violations:
            if v.severity == "error":
                base_score -= 20
            elif v.severity == "warning":
                base_score -= 10

        return RuleJudgeResult(
            is_compliant=all(v.severity != "error" for v in violations),
            violations=violations,
            compliance_score=max(0.0, base_score),
        )

    def evaluate_conversation(
        self, dialogue_history: list[dict[str, object]]
    ) -> RuleJudgeResult:
        """Evaluate entire conversation.

        Args:
            dialogue_history: List of conversation turns

        Returns:
            Combined evaluation result
        """
        all_violations: list[ConstraintViolation] = []
        total_score = 0.0
        agent_turns = [t for t in dialogue_history if t["role"] == "agent"]

        for turn in agent_turns:
            result = self.evaluate_message(turn["content"])
            all_violations.extend(result.violations)
            total_score += result.compliance_score

        avg_score = total_score / len(agent_turns) if agent_turns else 100.0

        return RuleJudgeResult(
            is_compliant=all(v.severity != "error" for v in all_violations),
            violations=all_violations,
            compliance_score=avg_score,
        )