"""Experiment runner for A/B testing."""

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class ExperimentConfig(BaseModel):
    """Configuration for an experiment.

    New shape (v2): single task per side + difficulty matrix.
    Old fields (baseline_version / candidate_version / task_ids / num_scenarios_per_task)
    are kept for backward compatibility with callers that haven't been updated.
    """

    experiment_id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""

    # New (v2)
    baseline_task_id: str = ""
    candidate_task_id: str = ""
    difficulty_levels: list[str] = Field(
        default_factory=lambda: ["easy", "medium", "hard"]
    )
    scenarios_per_level: int = 10

    # Legacy aliases (kept for back-compat with older callers)
    baseline_version: str = ""
    candidate_version: str = ""
    task_ids: list[str] = Field(default_factory=list)
    num_scenarios_per_task: int = 0

    def resolved_baseline_task_id(self) -> str:
        return self.baseline_task_id or self.baseline_version or (
            self.task_ids[0] if self.task_ids else ""
        )

    def resolved_candidate_task_id(self) -> str:
        return self.candidate_task_id or self.candidate_version or (
            self.task_ids[1] if len(self.task_ids) > 1 else (self.task_ids[0] if self.task_ids else "")
        )

    def resolved_difficulty_levels(self) -> list[str]:
        return self.difficulty_levels or ["easy", "medium", "hard"]

    def resolved_scenarios_per_level(self) -> int:
        if self.scenarios_per_level > 0:
            return self.scenarios_per_level
        return self.num_scenarios_per_task or 5


class LevelAggregate(BaseModel):
    """Aggregated metrics for a single difficulty level within a run."""

    difficulty: str
    total_samples: int = 0
    passed_samples: int = 0
    success_rate: float = 0.0
    avg_score: float = 0.0
    avg_cost: float = 0.0


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

    # v2 additions
    difficulty_levels: list[str] = Field(default_factory=list)
    results_by_difficulty: dict[str, list[dict]] = Field(default_factory=dict)
    per_level_aggregates: dict[str, LevelAggregate] = Field(default_factory=dict)


class ExperimentRunner:
    """Runs A/B experiments for version comparison."""

    def __init__(self, benchmark_runner=None, scenario_generator=None, task_loader=None):
        """Initialize the runner.

        Args:
            benchmark_runner: Benchmark runner for evaluation (optional)
            scenario_generator: ScenarioGenerator instance (lazy-created if None)
            task_loader: TaskLoader instance (lazy-created if None)
        """
        self.benchmark_runner = benchmark_runner
        self._scenario_generator = scenario_generator
        self._task_loader = task_loader

    @property
    def scenario_generator(self):
        if self._scenario_generator is None:
            from outbound_eval.scenarios.generator import ScenarioGenerator
            self._scenario_generator = ScenarioGenerator()
        return self._scenario_generator

    @property
    def task_loader(self):
        if self._task_loader is None:
            from outbound_eval.dataset.loader import TaskLoader
            self._task_loader = TaskLoader()
        return self._task_loader

    def run_experiment(
        self,
        config: ExperimentConfig,
        agent_factory: callable,
        eval_pipeline: callable,
    ) -> tuple[ExperimentRun, ExperimentRun]:
        """Run an experiment comparing two configurations.

        Args:
            config: Experiment configuration
            agent_factory: Factory function that creates agent given version
            eval_pipeline: Pipeline function for evaluation, signature
                ``eval_pipeline(agent, task_id, level) -> dict`` (the level
                argument is new in v2; older callers ignoring it are still OK).

        Returns:
            Tuple of (baseline_run, candidate_run)
        """
        baseline_task = config.resolved_baseline_task_id()
        candidate_task = config.resolved_candidate_task_id()

        baseline_run = self._run_version(
            config, "baseline", baseline_task, agent_factory, eval_pipeline
        )
        candidate_run = self._run_version(
            config, "candidate", candidate_task, agent_factory, eval_pipeline
        )
        return baseline_run, candidate_run

    def _run_version(
        self,
        config: ExperimentConfig,
        version: str,
        task_id: str,
        agent_factory: callable,
        eval_pipeline: callable,
    ) -> ExperimentRun:
        """Run evaluation for a single version, sliced by difficulty levels."""
        from outbound_eval.dataset.task import DifficultyLevel

        run_id = f"{config.experiment_id}_{version}"
        levels = config.resolved_difficulty_levels()
        scenarios_per_level = config.resolved_scenarios_per_level()

        results_by_diff: dict[str, list[dict]] = {lvl: [] for lvl in levels}

        agent = agent_factory(version)

        for level in levels:
            # Clone task and lock difficulty to this level
            try:
                task = self.task_loader.load(task_id).model_copy(deep=True)
            except Exception as e:
                print(f"[{run_id}] task load failed for {task_id}: {e}")
                continue

            try:
                task.difficulty = DifficultyLevel(level)
            except ValueError:
                print(f"[{run_id}] invalid difficulty level: {level}")
                continue

            # Generate scenarios locked to this level
            try:
                scenarios = self.scenario_generator.generate(
                    task,
                    num_scenarios=scenarios_per_level,
                    difficulty_distribution={level: 1.0},
                )
            except Exception as e:
                print(f"[{run_id}] scenario generation failed at {level}: {e}")
                scenarios = []

            for _sc in scenarios:
                try:
                    result = eval_pipeline(agent, task_id, level)
                except TypeError:
                    # Backward-compat: older eval_pipeline may not accept level
                    result = eval_pipeline(agent, task_id)
                # Ensure difficulty tag is present
                if isinstance(result, dict):
                    result.setdefault("difficulty", level)
                results_by_diff[level].append(result or {})

        per_level: dict[str, LevelAggregate] = {
            lvl: self._aggregate_level(lvl, results_by_diff.get(lvl, []))
            for lvl in levels
        }

        # Overall aggregates from all per-level results
        all_results = [r for rs in results_by_diff.values() for r in rs]
        total = len(all_results)
        passed = sum(1 for r in all_results if r.get("passed", False))
        scores = [r.get("overall_score", 0.0) for r in all_results if "overall_score" in r]
        costs = [r.get("cost", 0.0) for r in all_results if "cost" in r]

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
            difficulty_levels=list(levels),
            results_by_difficulty=results_by_diff,
            per_level_aggregates=per_level,
        )

    @staticmethod
    def _aggregate_level(level: str, results: list[dict]) -> LevelAggregate:
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        scores = [r.get("overall_score", 0.0) for r in results if "overall_score" in r]
        costs = [r.get("cost", 0.0) for r in results if "cost" in r]
        return LevelAggregate(
            difficulty=level,
            total_samples=total,
            passed_samples=passed,
            success_rate=(passed / total * 100) if total > 0 else 0.0,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            avg_cost=sum(costs) / len(costs) if costs else 0.0,
        )
