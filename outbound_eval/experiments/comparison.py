"""Experiment comparison with statistical analysis."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from outbound_eval.experiments.runner import ExperimentRun
from outbound_eval.analytics.bootstrap import BootstrapCI


class ExperimentComparison(BaseModel):
    """Comparison result between two experiment runs."""

    experiment_id: str = ""
    baseline_run: ExperimentRun
    candidate_run: ExperimentRun

    # Point estimates
    success_rate_delta: float = 0.0
    score_delta: float = 0.0
    cost_delta: float = 0.0

    # Bootstrap CI
    success_rate_delta_ci_95: tuple[float, float] = (0.0, 0.0)
    score_delta_ci_95: tuple[float, float] = (0.0, 0.0)

    # Statistical test
    p_value: float = 1.0
    is_significant: bool = False
    effect_size: float = 0.0

    # Recommendation
    recommendation: Literal["adopt_candidate", "keep_baseline", "inconclusive"] = "inconclusive"
    confidence: float = 0.0
    reasoning: str = ""


class ExperimentComparisonAnalyzer:
    """Analyzes experiment comparison results."""

    def __init__(self):
        """Initialize the analyzer."""
        self.bootstrap = BootstrapCI()

    def analyze(
        self,
        baseline_run: ExperimentRun,
        candidate_run: ExperimentRun,
        experiment_id: str = "",
    ) -> ExperimentComparison:
        """Analyze and compare two experiment runs.

        Args:
            baseline_run: Baseline version results
            candidate_run: Candidate version results
            experiment_id: Experiment ID

        Returns:
            Comparison result
        """
        # Point estimates
        success_rate_delta = candidate_run.success_rate - baseline_run.success_rate
        score_delta = candidate_run.avg_score - baseline_run.avg_score
        cost_delta = candidate_run.avg_cost - baseline_run.avg_cost

        # Bootstrap CI for success rate
        # Simulate individual results for bootstrap
        baseline_results = self._simulate_results(
            baseline_run.total_samples,
            baseline_run.success_rate / 100,
        )
        candidate_results = self._simulate_results(
            candidate_run.total_samples,
            candidate_run.success_rate / 100,
        )

        comparison = self.bootstrap.compare_groups(
            baseline_results, candidate_results, "rate_diff"
        )

        success_rate_ci = (comparison.ci_95_lower, comparison.ci_95_upper)

        # Estimate p-value (simplified)
        import math

        n1 = baseline_run.total_samples
        n2 = candidate_run.total_samples
        p1 = baseline_run.success_rate / 100
        p2 = candidate_run.success_rate / 100

        if n1 > 0 and n2 > 0:
            pooled_p = (baseline_run.passed_samples + candidate_run.passed_samples) / (n1 + n2)
            se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))

            if se > 0:
                z = abs(p1 - p2) / se
                # Approximate p-value using normal distribution
                p_value = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
            else:
                p_value = 1.0
        else:
            p_value = 1.0

        # Effect size (Cohen's h)
        def phi(p):
            return 2 * math.asin(math.sqrt(max(0.001, min(0.999, p))))

        if p1 > 0 and p2 > 0:
            effect_size = abs(phi(p2) - phi(p1))
        else:
            effect_size = 0.0

        # Recommendation
        if comparison.significant and success_rate_delta > 0:
            recommendation: Literal["adopt_candidate", "keep_baseline", "inconclusive"] = "adopt_candidate"
            confidence = 0.9 if p_value < 0.01 else 0.7
        elif comparison.significant and success_rate_delta < 0:
            recommendation = "keep_baseline"
            confidence = 0.9 if p_value < 0.01 else 0.7
        else:
            recommendation = "inconclusive"
            confidence = 0.5

        # Reasoning
        reasoning = self._generate_reasoning(
            success_rate_delta, p_value, effect_size, comparison.significant
        )

        return ExperimentComparison(
            experiment_id=experiment_id,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            success_rate_delta=success_rate_delta,
            score_delta=score_delta,
            cost_delta=cost_delta,
            success_rate_delta_ci_95=success_rate_ci,
            p_value=p_value,
            is_significant=comparison.significant,
            effect_size=effect_size,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _simulate_results(self, n_samples: int, success_rate: float) -> list[float]:
        """Simulate binary results for bootstrap."""
        import random

        return [1.0 if random.random() < success_rate else 0.0 for _ in range(n_samples)]

    def _generate_reasoning(
        self,
        delta: float,
        p_value: float,
        effect_size: float,
        significant: bool,
    ) -> str:
        """Generate reasoning text."""
        if not significant:
            return "差异不显著，无法得出确定结论。"

        if delta > 0:
            if p_value < 0.01:
                return f"候选版本显著优于基线版本（p={p_value:.3f}），建议采用。"
            else:
                return f"候选版本优于基线版本（p={p_value:.3f}），建议考虑采用。"
        else:
            if p_value < 0.01:
                return f"候选版本显著差于基线版本（p={p_value:.3f}），建议保持基线版本。"
            else:
                return f"候选版本差于基线版本（p={p_value:.3f}），建议保持基线版本。"