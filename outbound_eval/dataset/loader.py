"""Task loader utilities."""

import yaml
from pathlib import Path
from typing import Optional
from outbound_eval.dataset.task import EvaluationTask


class TaskLoader:
    """Loader for evaluation tasks."""

    def __init__(self, tasks_dir: Optional[str] = None):
        """Initialize the loader.

        Args:
            tasks_dir: Directory containing task YAML files.
                      Defaults to data/benchmarks/tasks/
        """
        if tasks_dir:
            self.tasks_dir = Path(tasks_dir)
        else:
            # /path/to/outbound_eval/outbound_eval/dataset/loader.py
            # → /path/to/outbound_eval/data/benchmarks/tasks
            package_dir = Path(__file__).parent.parent  # outbound_eval/
            self.tasks_dir = package_dir.parent / "data" / "benchmarks" / "tasks"

    def load(self, task_id: str) -> EvaluationTask:
        """Load a single task by ID.

        Args:
            task_id: The task ID (filename without .yaml extension)

        Returns:
            The loaded EvaluationTask

        Raises:
            FileNotFoundError: If task file doesn't exist
            ValueError: If task file is invalid
        """
        task_file = self.tasks_dir / f"{task_id}.yaml"
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")

        with open(task_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty task file: {task_file}")

        return EvaluationTask(**data)

    def load_all(self) -> list[EvaluationTask]:
        """Load all tasks from the tasks directory.

        Returns:
            List of all loaded EvaluationTask objects
        """
        tasks = []
        if not self.tasks_dir.exists():
            return tasks

        for task_file in self.tasks_dir.glob("*.yaml"):
            try:
                task = self.load(task_file.stem)
                tasks.append(task)
            except Exception as e:
                # Skip invalid task files
                print(f"Warning: Failed to load {task_file}: {e}")
                continue

        return tasks

    def save(self, task: EvaluationTask) -> None:
        """Save a task to YAML file.

        Args:
            task: The task to save
        """
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        task_file = self.tasks_dir / f"{task.task_id}.yaml"

        with open(task_file, "w", encoding="utf-8") as f:
            yaml.dump(
                task.model_dump(),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )