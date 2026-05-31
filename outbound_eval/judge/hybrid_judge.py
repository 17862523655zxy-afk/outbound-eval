"""Hybrid judge combining rule and LLM evaluation."""

from typing import Literal
from pydantic import BaseModel, Field
from outbound_eval.judge.rule_judge import RuleJudge
from outbound_eval.judge.llm_judge import LLMJudge, LLMJudgeResult
from outbound_eval.judge.rule_judge import RuleJudgeResult


class HybridJudgeResult(BaseModel):
    """Result of hybrid evaluation."""

    metric_name: str
    score: float
    judge_mode: Literal["rule", "llm", "hybrid", "fallback"]
    rule_score: float = 0.0
    llm_score: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)


class HybridJudge:
    """Hybrid judge combining rule-based and LLM-based evaluation."""

    def __init__(
        self,
        rule_judge: RuleJudge,
        llm_judge: LLMJudge,
        confidence_threshold: float = 0.7,
    ):
        """Initialize the hybrid judge.

        Args:
            rule_judge: Rule-based judge
            llm_judge: LLM-based judge
            confidence_threshold: Threshold for accepting rule-based result
        """
        self.rule_judge = rule_judge
        self.llm_judge = llm_judge
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        metric_name: str,
        dialogue_history: list[dict],
        rule_applicable: bool = True,
        llm_required: bool = False,
    ) -> HybridJudgeResult:
        """Evaluate with automatic mode selection.

        Args:
            metric_name: Name of the metric to evaluate
            dialogue_history: Conversation history
            rule_applicable: Whether rule-based evaluation is applicable
            llm_required: Whether LLM evaluation is required

        Returns:
            Hybrid evaluation result
        """
        rule_result: RuleJudgeResult | None = None
        llm_result: LLMJudgeResult | None = None

        # Rule-based evaluation
        if rule_applicable:
            rule_result = self.rule_judge.evaluate_conversation(dialogue_history)

        # LLM-based evaluation
        if llm_required or not rule_applicable:
            if metric_name == "naturalness":
                llm_result = self.llm_judge.evaluate_naturalness(dialogue_history)
            elif metric_name == "recovery":
                llm_result = self.llm_judge.evaluate_recovery(
                    {"context": dialogue_history}, ""
                )

        # Determine mode and score
        return self._determine_result(
            metric_name, rule_result, llm_result, llm_required
        )

    def _determine_result(
        self,
        metric_name: str,
        rule_result: RuleJudgeResult | None,
        llm_result: LLMJudgeResult | None,
        llm_required: bool,
    ) -> HybridJudgeResult:
        """Determine the final result based on available evaluations.

        Args:
            metric_name: Metric name
            rule_result: Rule-based result
            llm_result: LLM-based result
            llm_required: Whether LLM is required

        Returns:
            Final hybrid result
        """
        # If LLM is required, use LLM result
        if llm_required and llm_result:
            return HybridJudgeResult(
                metric_name=metric_name,
                score=llm_result.score,
                judge_mode="llm",
                rule_score=rule_result.compliance_score if rule_result else 0.0,
                llm_score=llm_result.score,
                confidence=llm_result.confidence,
                reasoning=llm_result.reasoning,
                evidence=llm_result.evidence,
            )

        # If rule result is available and confident, use it
        if rule_result and rule_result.is_compliant:
            return HybridJudgeResult(
                metric_name=metric_name,
                score=rule_result.compliance_score,
                judge_mode="rule",
                rule_score=rule_result.compliance_score,
                llm_score=llm_result.score if llm_result else 0.0,
                confidence=0.95,  # High confidence for rule-based results
                reasoning="Rule-based evaluation passed all constraints",
                evidence=[
                    f"No violations found" if not rule_result.violations
                    else f"Found {len(rule_result.violations)} minor violations"
                ],
            )

        # Fallback to LLM if rule failed
        if llm_result:
            return HybridJudgeResult(
                metric_name=metric_name,
                score=llm_result.score,
                judge_mode="fallback",
                rule_score=rule_result.compliance_score if rule_result else 0.0,
                llm_score=llm_result.score,
                confidence=llm_result.confidence,
                reasoning=llm_result.reasoning,
                evidence=llm_result.evidence,
            )

        # Default fallback
        return HybridJudgeResult(
            metric_name=metric_name,
            score=rule_result.compliance_score if rule_result else 75.0,
            judge_mode="rule",
            rule_score=rule_result.compliance_score if rule_result else 75.0,
            confidence=0.5,
            reasoning="No specific evaluation available",
            evidence=[],
        )