"""Data store for evaluation tasks and results."""

import json
from pathlib import Path
from typing import Optional, Any
from datetime import datetime


class DataStore:
    """Simple JSON-based data store."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the store.

        Args:
            data_dir: Root data directory. Defaults to ./data/
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path("./data")

        self.results_dir = self.data_dir / "results" / "raw"
        self.experiments_dir = self.data_dir / "experiments"

    def save_result(
        self, task_id: str, scenario_id: str, result: dict[str, Any]
    ) -> Path:
        """Save an evaluation result.

        Args:
            task_id: Task identifier
            scenario_id: Scenario identifier
            result: Result data

        Returns:
            Path to saved file
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{task_id}_{scenario_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return filepath

    def load_result(self, filepath: Path) -> dict[str, Any]:
        """Load a result from file.

        Args:
            filepath: Path to result file

        Returns:
            Result data
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_results(self, task_id: Optional[str] = None) -> list[Path]:
        """List all result files.

        Args:
            task_id: Optional filter by task ID

        Returns:
            List of result file paths
        """
        if not self.results_dir.exists():
            return []

        results = list(self.results_dir.glob("*.json"))

        if task_id:
            results = [r for r in results if r.name.startswith(task_id)]

        return sorted(results, key=lambda x: x.stat().st_mtime, reverse=True)

    def load_result_by_scenario(
        self, task_id: str, scenario_id: str
    ) -> Optional[dict[str, Any]]:
        """Load result by task and scenario ID.

        Args:
            task_id: Task identifier
            scenario_id: Scenario identifier

        Returns:
            Result data or None if not found
        """
        results = self.list_results(task_id)
        for r in results:
            if scenario_id in r.name:
                return self.load_result(r)
        return None

    def save_experiment(
        self, experiment_id: str, version: str, results: list[dict[str, Any]]
    ) -> Path:
        """Save experiment results.

        Args:
            experiment_id: Experiment identifier
            version: Version name (baseline/candidate)
            results: List of evaluation results

        Returns:
            Path to saved file
        """
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = exp_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_id": experiment_id,
                    "version": version,
                    "timestamp": datetime.now().isoformat(),
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return filepath

    def load_experiment(self, filepath: Path) -> dict[str, Any]:
        """Load experiment results.

        Args:
            filepath: Path to experiment file

        Returns:
            Experiment data
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)