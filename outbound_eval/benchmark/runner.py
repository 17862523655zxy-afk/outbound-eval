"""Benchmark runner."""

from typing import Optional
from outbound_eval.dataset.task import EvaluationTask
from outbound_eval.benchmark.pipeline import EvalPipeline
from outbound_eval.dataset.store import DataStore


class BenchmarkRunner:
    """Main benchmark runner for evaluation tasks."""

    def __init__(self):
        """Initialize the runner."""
        self.pipeline = EvalPipeline()
        self.store = DataStore()

    def run_task(
        self,
        task: EvaluationTask,
        num_scenarios: int = 10,
    ) -> list[dict]:
        """Run evaluation for a task.

        Args:
            task: Evaluation task
            num_scenarios: Number of scenarios to generate

        Returns:
            List of evaluation results
        """
        from outbound_eval.scenarios.generator import ScenarioGenerator

        generator = ScenarioGenerator()
        scenarios = generator.generate(task, num_scenarios=num_scenarios)

        results = self.pipeline.run(task, scenarios)

        # Save results
        for result in results:
            self.store.save_result(
                task.task_id,
                result.get("scenario_id", "unknown"),
                result,
            )

        return results

    def run_batch(
        self,
        tasks: list[EvaluationTask],
        num_scenarios: int = 10,
    ) -> dict:
        """Run batch evaluation for multiple tasks.

        Args:
            tasks: List of tasks
            num_scenarios: Scenarios per task

        Returns:
            Summary dict
        """
        all_results = {}

        for task in tasks:
            results = self.run_task(task, num_scenarios)
            all_results[task.task_id] = results

        # Calculate summary
        total = sum(len(r) for r in all_results.values())
        passed = sum(sum(1 for res in r if res.get("passed", False)) for r in all_results.values())
        success_rate = (passed / total * 100) if total > 0 else 0

        return {
            "total_evaluations": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": success_rate,
            "results_by_task": {
                task_id: {
                    "total": len(results),
                    "passed": sum(1 for r in results if r.get("passed", False)),
                }
                for task_id, results in all_results.items()
            },
        }