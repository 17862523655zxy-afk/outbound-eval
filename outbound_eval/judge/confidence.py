"""Judge confidence estimation."""

from typing import Literal
from pydantic import BaseModel, Field


class JudgeConfidence(BaseModel):
    """Confidence estimation for judge results."""

    confidence: float = Field(description="0.0-1.0 confidence level")
    confidence_level: Literal["high", "medium", "low"] = "medium"

    # Confidence factors
    factors: dict[str, float] = Field(
        default_factory=dict, description="Individual factor scores"
    )

    # Low confidence reasons
    low_confidence_reasons: list[str] = Field(default_factory=list)


class ConfidenceEstimator:
    """Estimates confidence for judge evaluations."""

    def estimate_from_response(self, response: dict) -> JudgeConfidence:
        """Estimate confidence from LLM response.

        Args:
            response: LLM response dict

        Returns:
            Confidence estimation
        """
        factors = {}
        reasons = []

        # Check evidence strength
        evidence = response.get("evidence", [])
        evidence_strength = min(1.0, len(evidence) / 3.0)
        factors["evidence_strength"] = evidence_strength

        if evidence_strength < 0.5:
            reasons.append("Insufficient evidence provided")

        # Check reasoning clarity
        reasoning = response.get("reasoning", "")
        if len(reasoning) > 50:
            factors["clarity"] = 0.8
        elif len(reasoning) > 20:
            factors["clarity"] = 0.6
        else:
            factors["clarity"] = 0.4
            reasons.append("Reasoning too brief")

        # Check score extremity (extreme scores may be less confident)
        score = response.get("score", 75.0)
        if 30 <= score <= 70:
            factors["certainty"] = 0.9  # Middle scores are more certain
        elif 20 <= score <= 80:
            factors["certainty"] = 0.7
        else:
            factors["certainty"] = 0.5
            reasons.append("Extreme score may indicate uncertainty")

        # Check for hedging language
        hedging_words = ["可能", "也许", "不确定", "maybe", "perhaps", "uncertain"]
        if any(word in reasoning.lower() for word in hedging_words):
            factors["hedging"] = 0.3
            reasons.append("Response contains hedging language")
        else:
            factors["hedging"] = 0.9

        # Calculate overall confidence
        weights = {
            "evidence_strength": 0.3,
            "clarity": 0.2,
            "certainty": 0.3,
            "hedging": 0.2,
        }

        overall = sum(factors.get(k, 0) * v for k, v in weights.items())

        # Determine level
        if overall >= 0.8:
            level: Literal["high", "medium", "low"] = "high"
        elif overall >= 0.5:
            level = "medium"
        else:
            level = "low"

        return JudgeConfidence(
            confidence=overall,
            confidence_level=level,
            factors=factors,
            low_confidence_reasons=reasons,
        )

    def estimate_from_context(
        self, dialogue_length: int, metric_complexity: str
    ) -> JudgeConfidence:
        """Estimate confidence based on evaluation context.

        Args:
            dialogue_length: Number of turns in dialogue
            metric_complexity: Complexity level of metric

        Returns:
            Confidence estimation
        """
        factors = {}

        # Longer dialogues provide more context
        if dialogue_length >= 5:
            factors["context_adequacy"] = 0.9
        elif dialogue_length >= 3:
            factors["context_adequacy"] = 0.7
        else:
            factors["context_adequacy"] = 0.5

        # Complex metrics are harder to judge
        if metric_complexity == "simple":
            factors["metric_complexity"] = 0.9
        elif metric_complexity == "medium":
            factors["metric_complexity"] = 0.7
        else:
            factors["metric_complexity"] = 0.5

        weights = {"context_adequacy": 0.6, "metric_complexity": 0.4}
        overall = sum(factors.get(k, 0) * v for k, v in weights.items())

        if overall >= 0.8:
            level: Literal["high", "medium", "low"] = "high"
        elif overall >= 0.5:
            level = "medium"
        else:
            level = "low"

        return JudgeConfidence(
            confidence=overall,
            confidence_level=level,
            factors=factors,
            low_confidence_reasons=[],
        )