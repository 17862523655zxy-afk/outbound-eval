"""Difficulty stratification analysis."""

from typing import Optional
from pydantic import BaseModel, Field
from outbound_eval.dataset.task import DifficultyLevel
from outbound_eval.analytics.bootstrap import BootstrapCI


class DifficultyMetrics(BaseModel):
    """Metrics for a difficulty level."""

    difficulty: str
    total_cases: int = 0
    passed_cases: int = 0
    success_rate: float = 0.0
    avg_score: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    failure_distribution: dict[str, int] = Field(default_factory=dict)


class DifficultyReport(BaseModel):
    """Report for difficulty stratification."""

    task_id: str
    evaluation_time: str
    difficulty_metrics: list[DifficultyMetrics]
    easy_success_rate: float = 0.0
    medium_success_rate: float = 0.0
    hard_success_rate: float = 0.0
    difficulty_curve: list[dict] = Field(default_factory=list)


class DifficultyAnalyzer:
    """Analyzes performance by difficulty level."""

    def __init__(self):
        """Initialize the analyzer."""
        self.bootstrap = BootstrapCI()

    def analyze(self, eval_results: list[dict]) -> DifficultyReport:
        """Analyze evaluation results by difficulty.

        Args:
            eval_results: List of evaluation results

        Returns:
            Difficulty report
        """
        # Group by difficulty
        difficulty_groups: dict[str, list[dict]] = {
            "easy": [],
            "medium": [],
            "hard": [],
        }

        for result in eval_results:
            difficulty = result.get("difficulty", "medium")
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(result)

        difficulty_metrics: list[DifficultyMetrics] = []

        for diff_level, results in difficulty_groups.items():
            if not results:
                continue

            total = len(results)
            passed = sum(1 for r in results if r.get("passed", False))
            scores = [r.get("overall_score", 0.0) for r in results]

            # Bootstrap CI
            success_flags = [1 if r.get("passed", False) else 0 for r in results]
            ci_result = self.bootstrap.calculate_ci(success_flags, "rate")

            # Failure distribution
            failure_dist: dict[str, int] = {}
            for r in results:
                if not r.get("passed", False):
                    for reason in r.get("failure_reasons", []):
                        failure_dist[reason] = failure_dist.get(reason, 0) + 1

            metrics = DifficultyMetrics(
                difficulty=diff_level,
                total_cases=total,
                passed_cases=passed,
                success_rate=(passed / total * 100) if total > 0 else 0.0,
                avg_score=sum(scores) / len(scores) if scores else 0.0,
                ci_95_lower=ci_result.ci_95_lower * 100,
                ci_95_upper=ci_result.ci_95_upper * 100,
                failure_distribution=failure_dist,
            )
            difficulty_metrics.append(metrics)

        # Build difficulty curve
        difficulty_curve = [
            {"difficulty": m.difficulty, "success_rate": m.success_rate}
            for m in difficulty_metrics
        ]

        # Get rates by level
        def get_rate(level: str) -> float:
            for m in difficulty_metrics:
                if m.difficulty == level:
                    return m.success_rate
            return 0.0

        from datetime import datetime

        return DifficultyReport(
            task_id=eval_results[0].get("task_id", "unknown") if eval_results else "unknown",
            evaluation_time=datetime.now().isoformat(),
            difficulty_metrics=difficulty_metrics,
            easy_success_rate=get_rate("easy"),
            medium_success_rate=get_rate("medium"),
            hard_success_rate=get_rate("hard"),
            difficulty_curve=difficulty_curve,
        )