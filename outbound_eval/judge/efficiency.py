"""Efficiency metrics evaluation."""

from typing import Optional
from pydantic import BaseModel, Field
from outbound_eval.infra.config import settings


class EfficiencyMetrics(BaseModel):
    """Efficiency metrics for a call."""

    # Conversation efficiency
    total_turns: int = Field(default=0, description="Total conversation turns")
    avg_turns: float = Field(default=0.0, description="Average turns")
    median_turns: float = Field(default=0.0, description="Median turns")

    # Token efficiency
    total_tokens: int = Field(default=0, description="Total tokens consumed")
    avg_tokens_per_call: float = Field(default=0.0, description="Average tokens per call")
    cost_per_call: float = Field(default=0.0, description="Cost per call in $")

    # Time efficiency
    total_duration_seconds: float = Field(default=0.0, description="Total duration")
    avg_duration_seconds: float = Field(default=0.0, description="Average duration")

    # Overall efficiency
    efficiency_score: float = Field(default=0.0, description="0-100 efficiency score")

    # Comparison with target
    turns_vs_target: float = Field(default=0.0, description="Turns compared to target")
    cost_vs_target: float = Field(default=0.0, description="Cost compared to target")


class EfficiencyJudge:
    """Evaluates efficiency metrics."""

    def __init__(
        self,
        target_turns: Optional[float] = None,
        target_cost: Optional[float] = None,
        token_price: Optional[float] = None,
    ):
        """Initialize the judge.

        Args:
            target_turns: Target turns per call
            target_cost: Target cost per success
            token_price: Token price per 1K tokens
        """
        self.target_turns = target_turns or settings.target_turns
        self.target_cost = target_cost or settings.target_cost_per_success
        self.token_price = token_price or settings.token_price_per_1k

    def evaluate(
        self,
        dialogue_history: list[dict[str, object]],
        total_tokens: int = 0,
    ) -> EfficiencyMetrics:
        """Evaluate efficiency.

        Args:
            dialogue_history: Conversation history
            total_tokens: Total tokens consumed

        Returns:
            Efficiency metrics
        """
        total_turns = len(dialogue_history)

        # Calculate costs
        cost_per_call = (total_tokens / 1000) * self.token_price

        # Calculate efficiency score
        turns_score = min(100.0, self.target_turns / max(total_turns, 1) * 100)
        cost_score = min(100.0, self.target_cost / max(cost_per_call, 0.001) * 100)

        efficiency_score = (turns_score * 0.5 + cost_score * 0.5)

        return EfficiencyMetrics(
            total_turns=total_turns,
            avg_turns=float(total_turns),  # For batch, this would be averaged
            median_turns=float(total_turns),
            total_tokens=total_tokens,
            avg_tokens_per_call=float(total_tokens),
            cost_per_call=cost_per_call,
            total_duration_seconds=total_turns * 15.0,  # ~15 seconds per turn estimate
            avg_duration_seconds=total_turns * 15.0,
            efficiency_score=efficiency_score,
            turns_vs_target=total_turns - self.target_turns,
            cost_vs_target=cost_per_call - self.target_cost,
        )

    def evaluate_batch(
        self, results: list[dict]
    ) -> EfficiencyMetrics:
        """Evaluate efficiency for a batch of calls.

        Args:
            results: List of evaluation results with dialogue history

        Returns:
            Aggregated efficiency metrics
        """
        import statistics

        total_turns_list = []
        total_tokens_list = []
        cost_list = []

        for result in results:
            dialogue = result.get("dialogue_history", [])
            tokens = result.get("total_tokens", 0)

            total_turns_list.append(len(dialogue))
            total_tokens_list.append(tokens)
            cost_list.append((tokens / 1000) * self.token_price)

        total_turns = sum(total_turns_list) if total_turns_list else 0
        total_tokens = sum(total_tokens_list) if total_tokens_list else 0
        avg_turns = statistics.mean(total_turns_list) if total_turns_list else 0.0
        median_turns = statistics.median(total_turns_list) if total_turns_list else 0.0
        avg_tokens = statistics.mean(total_tokens_list) if total_tokens_list else 0.0
        avg_cost = statistics.mean(cost_list) if cost_list else 0.0

        # Calculate efficiency
        turns_score = min(100.0, self.target_turns / max(avg_turns, 1) * 100)
        cost_score = min(100.0, self.target_cost / max(avg_cost, 0.001) * 100)

        efficiency_score = (turns_score * 0.5 + cost_score * 0.5)

        return EfficiencyMetrics(
            total_turns=total_turns,
            avg_turns=avg_turns,
            median_turns=median_turns,
            total_tokens=total_tokens,
            avg_tokens_per_call=avg_tokens,
            cost_per_call=avg_cost,
            total_duration_seconds=total_turns * 15.0,
            avg_duration_seconds=avg_turns * 15.0,
            efficiency_score=efficiency_score,
            turns_vs_target=avg_turns - self.target_turns,
            cost_vs_target=avg_cost - self.target_cost,
        )