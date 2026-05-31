"""Persona metrics analysis."""

from typing import Optional
from pydantic import BaseModel, Field
from outbound_eval.analytics.bootstrap import BootstrapCI


class PersonaMetrics(BaseModel):
    """Metrics for a single persona type."""

    persona_type: str
    total_cases: int = 0
    passed_cases: int = 0
    success_rate: float = 0.0
    avg_score: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    failure_distribution: dict[str, int] = Field(default_factory=dict)


class PersonaMetricsReport(BaseModel):
    """Report for all persona metrics."""

    task_id: str
    evaluation_time: str
    persona_metrics: list[PersonaMetrics]
    weakest_persona: str = ""
    strongest_persona: str = ""
    avg_success_rate: float = 0.0
    coverage_gap: float = 0.0
    improvement_priority: list[str] = Field(default_factory=list)


class PersonaMetricsAnalyzer:
    """Analyzes metrics by user persona type."""

    def __init__(self):
        """Initialize the analyzer."""
        self.bootstrap = BootstrapCI()

    def analyze(self, eval_results: list[dict]) -> PersonaMetricsReport:
        """Analyze evaluation results by persona.

        Args:
            eval_results: List of evaluation results with persona info

        Returns:
            Persona metrics report
        """
        # Group by persona type
        persona_groups: dict[str, list[dict]] = {}
        for result in eval_results:
            persona_type = result.get("persona_type", "unknown")
            if persona_type not in persona_groups:
                persona_groups[persona_type] = []
            persona_groups[persona_type].append(result)

        # Calculate metrics for each persona
        persona_metrics: list[PersonaMetrics] = []
        success_rates: dict[str, float] = {}

        for persona_type, results in persona_groups.items():
            total = len(results)
            passed = sum(1 for r in results if r.get("passed", False))
            scores = [r.get("overall_score", 0.0) for r in results]

            # Bootstrap CI for success rate
            success_flags = [1 if r.get("passed", False) else 0 for r in results]
            ci_result = self.bootstrap.calculate_ci(success_flags, "rate")

            # Get failure distribution
            failure_dist: dict[str, int] = {}
            for r in results:
                if not r.get("passed", False):
                    for reason in r.get("failure_reasons", []):
                        failure_dist[reason] = failure_dist.get(reason, 0) + 1

            metrics = PersonaMetrics(
                persona_type=persona_type,
                total_cases=total,
                passed_cases=passed,
                success_rate=(passed / total * 100) if total > 0 else 0.0,
                avg_score=sum(scores) / len(scores) if scores else 0.0,
                ci_95_lower=ci_result.ci_95_lower * 100,
                ci_95_upper=ci_result.ci_95_upper * 100,
                failure_distribution=failure_dist,
            )
            persona_metrics.append(metrics)
            success_rates[persona_type] = metrics.success_rate

        # Find weakest and strongest
        if success_rates:
            weakest = min(success_rates, key=success_rates.get)
            strongest = max(success_rates, key=success_rates.get)
            avg_rate = sum(success_rates.values()) / len(success_rates)
            gap = max(success_rates.values()) - min(success_rates.values())
        else:
            weakest = ""
            strongest = ""
            avg_rate = 0.0
            gap = 0.0

        # Generate improvement priority
        priorities = sorted(
            [(p.persona_type, p.success_rate) for p in persona_metrics],
            key=lambda x: x[1],
        )
        improvement_priority = [
            f"优先提升 {p[0]} 类型用户（当前成功率 {p[1]:.1f}%）"
            for p in priorities[:3]
        ]

        from datetime import datetime

        return PersonaMetricsReport(
            task_id=eval_results[0].get("task_id", "unknown") if eval_results else "unknown",
            evaluation_time=datetime.now().isoformat(),
            persona_metrics=persona_metrics,
            weakest_persona=weakest,
            strongest_persona=strongest,
            avg_success_rate=avg_rate,
            coverage_gap=gap,
            improvement_priority=improvement_priority,
        )