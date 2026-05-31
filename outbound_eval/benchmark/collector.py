"""Result collector."""

from typing import Optional
from outbound_eval.dataset.store import DataStore


class ResultCollector:
    """Collects and aggregates evaluation results."""

    def __init__(self, store: Optional[DataStore] = None):
        """Initialize the collector.

        Args:
            store: Data store instance
        """
        self.store = store or DataStore()

    def collect_all(self, task_id: Optional[str] = None) -> list[dict]:
        """Collect all results.

        Args:
            task_id: Optional filter by task ID

        Returns:
            List of results
        """
        result_files = self.store.list_results(task_id)
        return [self.store.load_result(f) for f in result_files]

    def aggregate_by_metric(self, results: list[dict]) -> dict:
        """Aggregate results by metric.

        Args:
            results: List of results

        Returns:
            Aggregated metrics
        """
        metrics = ["task_success", "flow_adherence", "state_tracking", "compliance", "recovery", "naturalness", "efficiency"]

        aggregated = {}
        for metric in metrics:
            scores = [r.get(metric, 0.0) for r in results if metric in r]
            if scores:
                aggregated[metric] = {
                    "avg": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }

        return aggregated

    def aggregate_by_persona(self, results: list[dict]) -> dict:
        """Aggregate results by persona type.

        Args:
            results: List of results

        Returns:
            Results by persona
        """
        persona_groups: dict[str, list[dict]] = {}

        for result in results:
            persona = result.get("persona_type", "unknown")
            if persona not in persona_groups:
                persona_groups[persona] = []
            persona_groups[persona].append(result)

        aggregated = {}
        for persona, persona_results in persona_groups.items():
            total = len(persona_results)
            passed = sum(1 for r in persona_results if r.get("passed", False))

            aggregated[persona] = {
                "total": total,
                "passed": passed,
                "success_rate": (passed / total * 100) if total > 0 else 0,
                "avg_score": sum(r.get("overall_score", 0) for r in persona_results) / total if total > 0 else 0,
            }

        return aggregated