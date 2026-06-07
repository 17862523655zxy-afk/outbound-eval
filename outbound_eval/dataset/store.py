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

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments with metadata.

        Returns:
            List of experiment metadata dicts.
        """
        experiments = []
        if not self.experiments_dir.exists():
            return experiments

        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                exp_id = exp_dir.name
                files = list(exp_dir.glob("*.json"))
                if files:
                    # Use latest file's mtime (epoch ms) for stable display
                    latest = max(files, key=lambda p: p.stat().st_mtime)
                    data = self.load_experiment(latest)
                    versions = [f.stem.split("_")[0] for f in files]
                    # Pull is_cross_business & task IDs from comparison if present
                    cmp_files = [f for f in files if f.stem.startswith("comparison")]
                    is_cross = None
                    baseline_task = ""
                    candidate_task = ""
                    if cmp_files:
                        cmp_data = self.load_experiment(cmp_files[0])
                        cmp_results = cmp_data.get("results", [])
                        if cmp_results and isinstance(cmp_results[0], dict):
                            is_cross = bool(cmp_results[0].get("is_cross_business", False))
                            baseline_task = cmp_results[0].get("baseline_task_id", "")
                            candidate_task = cmp_results[0].get("candidate_task_id", "")
                    experiments.append({
                        "experiment_id": exp_id,
                        "timestamp": data.get("timestamp", ""),
                        "timestamp_ms": int(latest.stat().st_mtime * 1000),
                        "versions": versions,
                        "is_cross_business": is_cross,
                        "baseline_task_id": baseline_task,
                        "candidate_task_id": candidate_task,
                        "result_count": sum(
                            len(self.load_experiment(f).get("results", []))
                            for f in files
                        ),
                    })
        return sorted(experiments, key=lambda x: x.get("timestamp_ms", 0), reverse=True)

    def load_experiment_versions(self, experiment_id: str) -> dict[str, dict[str, Any]]:
        """Load all versions of an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dict mapping version name to experiment data.
        """
        exp_dir = self.experiments_dir / experiment_id
        if not exp_dir.exists():
            return {}

        versions = {}
        for f in exp_dir.glob("*.json"):
            version = f.stem.split("_")[0]
            versions[version] = self.load_experiment(f)
        return versions

    def load_experiment(self, filepath: Path) -> dict[str, Any]:
        """Load experiment results.

        Args:
            filepath: Path to experiment file

        Returns:
            Experiment data
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)