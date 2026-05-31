"""Run monitor for real-time evaluation tracking."""

import threading
import time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RunState:
    """Real-time run state."""

    status: str = "pending"  # pending / running / completed / failed
    task_id: str = ""
    total_cases: int = 0
    completed_cases: int = 0
    current_case: str = ""
    success_count: int = 0
    total_score: float = 0.0
    total_turns: int = 0
    total_cost: float = 0.0
    logs: list[dict] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    persona_counts: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.completed_cases == 0:
            return 0.0
        return (self.success_count / self.completed_cases) * 100

    @property
    def avg_score(self) -> float:
        """Calculate average score."""
        if self.completed_cases == 0:
            return 0.0
        return self.total_score / self.completed_cases

    @property
    def avg_turns(self) -> float:
        """Calculate average turns."""
        if self.completed_cases == 0:
            return 0.0
        return self.total_turns / self.completed_cases

    @property
    def avg_cost(self) -> float:
        """Calculate average cost."""
        if self.completed_cases == 0:
            return 0.0
        return self.total_cost / self.completed_cases

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_cases == 0:
            return 0.0
        return (self.completed_cases / self.total_cases) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> dict:
        """Convert to dict for API response."""
        return {
            "status": self.status,
            "task_id": self.task_id,
            "total_cases": self.total_cases,
            "completed_cases": self.completed_cases,
            "current_case": self.current_case,
            "success_count": self.success_count,
            "success_rate": round(self.success_rate, 1),
            "avg_score": round(self.avg_score, 1),
            "avg_turns": round(self.avg_turns, 1),
            "avg_cost": round(self.avg_cost, 4),
            "progress_percent": round(self.progress_percent, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "logs": self.logs[-50:],  # Last 50 logs
            "persona_distribution": self.persona_counts,
        }

    def add_log(self, level: str, message: str) -> None:
        """Add a log entry."""
        self.logs.append({
            "timestamp": time.time(),
            "level": level,
            "message": message,
        })

    def reset(self) -> None:
        """Reset state."""
        self.status = "pending"
        self.task_id = ""
        self.total_cases = 0
        self.completed_cases = 0
        self.current_case = ""
        self.success_count = 0
        self.total_score = 0.0
        self.total_turns = 0
        self.total_cost = 0.0
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.persona_counts = {}


class RunMonitor:
    """Thread-safe run monitor singleton."""

    _instance: Optional["RunMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RunMonitor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._state = RunState()
        return cls._instance

    @property
    def state(self) -> RunState:
        """Get current state."""
        return self._state

    def start_run(self, task_id: str, total_cases: int) -> None:
        """Start a new run."""
        self._state.reset()
        self._state.status = "running"
        self._state.task_id = task_id
        self._state.total_cases = total_cases
        self._state.start_time = time.time()
        self._state.add_log("INFO", f"Run started for task: {task_id}")
        self._state.add_log("INFO", f"Total scenarios: {total_cases}")

    def scenario_generated(self, scenario_id: str) -> None:
        """Mark scenario as generated."""
        self._state.current_case = scenario_id
        self._state.add_log("INFO", f"Scenario generated: {scenario_id}")

    def scenario_started(self, scenario_id: str) -> None:
        """Mark scenario as started."""
        self._state.current_case = scenario_id
        self._state.add_log("INFO", f"Start simulation: {scenario_id}")

    def scenario_completed(
        self,
        scenario_id: str,
        passed: bool,
        score: float,
        turns: int,
        cost: float,
        persona_type: str = "",
    ) -> None:
        """Mark scenario as completed."""
        self._state.completed_cases += 1
        self._state.total_score += score
        self._state.total_turns += turns
        self._state.total_cost += cost

        if passed:
            self._state.success_count += 1
            self._state.add_log("PASS", f"Scenario {scenario_id} - Score: {score:.1f}")
        else:
            self._state.add_log("FAIL", f"Scenario {scenario_id} - Score: {score:.1f}")

        if persona_type:
            self._state.persona_counts[persona_type] = self._state.persona_counts.get(persona_type, 0) + 1

    def scenario_failed(self, scenario_id: str, error: str) -> None:
        """Mark scenario as failed."""
        self._state.completed_cases += 1
        self._state.add_log("ERROR", f"Scenario {scenario_id} failed: {error}")

    def event_injected(self, event_type: str, turn: int) -> None:
        """Log event injection."""
        self._state.add_log("EVENT", f"Injected {event_type} at turn {turn}")

    def judge_complete(self, scenario_id: str, score: float) -> None:
        """Log judge completion."""
        self._state.add_log("INFO", f"Judge complete for {scenario_id}: {score:.1f}")

    def finish_run(self) -> None:
        """Finish the run."""
        self._state.status = "completed"
        self._state.end_time = time.time()
        self._state.add_log("INFO", "Run completed")

    def fail_run(self, error: str) -> None:
        """Mark run as failed."""
        self._state.status = "failed"
        self._state.end_time = time.time()
        self._state.add_log("ERROR", f"Run failed: {error}")
