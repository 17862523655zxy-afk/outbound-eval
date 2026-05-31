"""Experiment runner for A/B testing."""

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class ExperimentConfig(BaseModel):
    """Configuration for an experiment."""

    experiment_id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    baseline_version: str = "v1"
    candidate_version: str = "v2"
    task_ids: list[str] = Field(default_factory=list)
    num_scenarios_per_task: int = 10


class ExperimentRun(BaseModel):
    """Result of a single experiment run."""

    run_id: str
    experiment_id: str
    version: str = ""
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    total_samples: int = 0
    passed_samples: int = 0
    success_rate: float = 0.0
    avg_score: float = 0.0
    avg_cost: float = 0.0
    completed_at: Optional[str] = None


class ExperimentRunner:
    """Runs A/B experiments for version comparison."""

    def __init__(self, benchmark_runner=None):
        """Initialize the runner.

        Args:
            benchmark_runner: Benchmark runner for evaluation
        """
        self.benchmark_runner = benchmark_runner

    def run_experiment(
        self,
        config: ExperimentConfig,
        agent_factory: callable,
        eval_pipeline: callable,
    ) -> tuple[ExperimentRun, ExperimentRun]:
        """Run an experiment comparing two versions.

        Args:
            config: Experiment configuration
            agent_factory: Factory function that creates agent given version
            eval_pipeline: Pipeline function for evaluation

        Returns:
            Tuple of (baseline_run, candidate_run)
        """
        # Run baseline version
        baseline_run = self._run_version(
            config, config.baseline_version, agent_factory, eval_pipeline
        )

        # Run candidate version
        candidate_run = self._run_version(
            config, config.candidate_version, agent_factory, eval_pipeline
        )

        return baseline_run, candidate_run

    def _run_version(
        self,
        config: ExperimentConfig,
        version: str,
        agent_factory: callable,
        eval_pipeline: callable,
    ) -> ExperimentRun:
        """Run evaluation for a single version.

        Args:
            config: Experiment configuration
            version: Version to test
            agent_factory: Factory for creating agent
            eval_pipeline: Evaluation pipeline

        Returns:
            Experiment run result
        """
        run_id = f"{config.experiment_id}_{version}"

        # Create agent
        agent = agent_factory(version)

        # Run evaluations
        results = []
        for task_id in config.task_ids:
            for i in range(config.num_scenarios_per_task):
                try:
                    result = eval_pipeline(agent, task_id)
                    results.append(result)
                except Exception as e:
                    print(f"Error in {run_id}: {e}")

        # Calculate metrics
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        scores = [r.get("overall_score", 0.0) for r in results]
        costs = [r.get("cost", 0.0) for r in results]

        return ExperimentRun(
            run_id=run_id,
            experiment_id=config.experiment_id,
            version=version,
            status="completed",
            total_samples=total,
            passed_samples=passed,
            success_rate=(passed / total * 100) if total > 0 else 0.0,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            avg_cost=sum(costs) / len(costs) if costs else 0.0,
            completed_at=datetime.now().isoformat(),
        )