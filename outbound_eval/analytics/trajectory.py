"""Conversation trajectory analysis for success patterns."""

from typing import Optional
from pydantic import BaseModel, Field
import hashlib


class TrajectoryStep(BaseModel):
    """A step in the conversation trajectory."""

    step_id: str
    step_name: str
    turn_range: tuple[int, int] = (0, 0)


class ConversationTrajectory(BaseModel):
    """Conversation trajectory record."""

    trajectory_id: str
    task_id: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    step_sequence: str = ""
    persona_type: str = ""
    difficulty: str = ""
    total_turns: int = 0
    outcome: str = "unknown"
    final_score: float = 0.0
    trajectory_hash: str = ""


class TrajectoryPattern(BaseModel):
    """A common trajectory pattern."""

    pattern_id: str
    pattern_name: str
    step_sequence: list[str] = Field(default_factory=list)
    description: str = ""
    occurrence_count: int = 0
    occurrence_rate: float = 0.0
    success_rate: float = 0.0
    avg_score: float = 0.0
    representative_case_id: Optional[str] = None


class TrajectoryReport(BaseModel):
    """Report for trajectory analysis."""

    task_id: str
    total_successful_cases: int = 0
    unique_trajectories: int = 0
    top_success_patterns: list[TrajectoryPattern] = Field(default_factory=list)
    failure_patterns: list[TrajectoryPattern] = Field(default_factory=list)


class SuccessPatternAnalyzer:
    """Analyzes success patterns in conversations."""

    def __init__(self, step_extractors: Optional[dict] = None):
        """Initialize the analyzer.

        Args:
            step_extractors: Dict mapping step_id to extraction functions
        """
        self.step_extractors = step_extractors or self._default_extractors()

    def _default_extractors(self) -> dict:
        """Get default step extractors."""
        return {
            "identity_confirm": lambda msg: "请问" in msg or "是吗" in msg,
            "contract_explain": lambda msg: "合同" in msg and "生效" in msg,
            "commitment_request": lambda msg: "开始" in msg or "配送" in msg,
            "commitment_obtain": lambda msg: any(
                kw in msg for kw in ["可以", "好的", "没问题"]
            ),
            "rejection": lambda msg: any(kw in msg for kw in ["不想", "不送了", "退出"]),
            "retention": lambda msg: any(kw in msg for kw in ["理解", "坚持", "挽留"]),
            "safety_reminder": lambda msg: "安全" in msg,
            "end": lambda msg: any(kw in msg for kw in ["谢谢", "再见", "联系"]),
        }

    def extract_trajectory(
        self, dialogue_history: list[dict], task_id: str = ""
    ) -> ConversationTrajectory:
        """Extract trajectory from dialogue history.

        Args:
            dialogue_history: Conversation history
            task_id: Task ID

        Returns:
            Extracted trajectory
        """
        steps: list[TrajectoryStep] = []
        turn = 0

        for i, turn_data in enumerate(dialogue_history):
            if turn_data["role"] != "agent":
                continue

            turn += 1
            msg = turn_data["content"]

            for step_id, extractor in self.step_extractors.items():
                if extractor(msg):
                    steps.append(
                        TrajectoryStep(
                            step_id=step_id,
                            step_name=step_id.replace("_", " "),
                            turn_range=(turn, turn),
                        )
                    )

        # Build sequence string
        sequence = " → ".join([s.step_id for s in steps])

        # Create hash for grouping
        hash_str = hashlib.md5(sequence.encode()).hexdigest()[:8]

        # Determine outcome
        user_msgs = [t["content"] for t in dialogue_history if t["role"] == "user"]
        all_text = " ".join(user_msgs)

        if any(kw in all_text for kw in ["可以", "好的", "没问题", "愿意"]):
            outcome = "success"
        elif any(kw in all_text for kw in ["不想", "不送了", "退出"]):
            outcome = "failure"
        else:
            outcome = "inconclusive"

        return ConversationTrajectory(
            trajectory_id=f"{task_id}_traj_{hash_str}",
            task_id=task_id,
            steps=steps,
            step_sequence=sequence,
            total_turns=len(dialogue_history),
            outcome=outcome,
            trajectory_hash=hash_str,
        )

    def analyze(
        self, eval_results: list[dict]
    ) -> TrajectoryReport:
        """Analyze success patterns.

        Args:
            eval_results: List of evaluation results

        Returns:
            Trajectory analysis report
        """
        successful_trajectories: list[ConversationTrajectory] = []
        all_trajectories: list[ConversationTrajectory] = []

        for result in eval_results:
            dialogue = result.get("dialogue_history", [])
            if not dialogue:
                continue

            trajectory = self.extract_trajectory(
                dialogue, result.get("task_id", "")
            )
            trajectory.persona_type = result.get("persona_type", "")
            trajectory.difficulty = result.get("difficulty", "")
            trajectory.final_score = result.get("overall_score", 0.0)

            all_trajectories.append(trajectory)

            if trajectory.outcome == "success":
                successful_trajectories.append(trajectory)

        # Group by trajectory hash
        trajectory_groups: dict[str, list[ConversationTrajectory]] = {}
        for traj in successful_trajectories:
            key = traj.trajectory_hash
            if key not in trajectory_groups:
                trajectory_groups[key] = []
            trajectory_groups[key].append(traj)

        # Build patterns
        patterns: list[TrajectoryPattern] = []
        total_successful = len(successful_trajectories)

        for hash_key, trajs in trajectory_groups.items():
            if len(trajs) < 2:  # At least 2 occurrences
                continue

            pattern = TrajectoryPattern(
                pattern_id=f"pattern_{hash_key}",
                pattern_name=self._generate_pattern_name(trajs[0]),
                step_sequence=trajs[0].step_sequence.split(" → "),
                description=self._generate_description(trajs[0]),
                occurrence_count=len(trajs),
                occurrence_rate=len(trajs) / total_successful if total_successful > 0 else 0,
                success_rate=100.0,  # All in this group are successful
                avg_score=sum(t.final_score for t in trajs) / len(trajs),
                representative_case_id=trajs[0].trajectory_id,
            )
            patterns.append(pattern)

        # Sort by occurrence rate
        patterns.sort(key=lambda p: p.occurrence_rate, reverse=True)

        # Build failure patterns
        failure_groups: dict[str, list[ConversationTrajectory]] = {}
        for traj in all_trajectories:
            if traj.outcome == "failure":
                key = traj.trajectory_hash
                if key not in failure_groups:
                    failure_groups[key] = []
                failure_groups[key].append(traj)

        failure_patterns: list[TrajectoryPattern] = []
        total_failed = sum(len(v) for v in failure_groups.values())

        for hash_key, trajs in failure_groups.items():
            pattern = TrajectoryPattern(
                pattern_id=f"fp_{hash_key}",
                pattern_name=self._generate_pattern_name(trajs[0]),
                step_sequence=trajs[0].step_sequence.split(" → ") if trajs[0].step_sequence else [],
                description="失败路径",
                occurrence_count=len(trajs),
                occurrence_rate=len(trajs) / total_failed if total_failed > 0 else 0,
                success_rate=0.0,
                avg_score=sum(t.final_score for t in trajs) / len(trajs),
            )
            failure_patterns.append(pattern)

        failure_patterns.sort(key=lambda p: p.occurrence_rate, reverse=True)

        return TrajectoryReport(
            task_id=eval_results[0].get("task_id", "unknown") if eval_results else "unknown",
            total_successful_cases=total_successful,
            unique_trajectories=len(trajectory_groups),
            top_success_patterns=patterns[:5],  # Top 5
            failure_patterns=failure_patterns[:3],  # Top 3 failures
        )

    def _generate_pattern_name(self, trajectory: ConversationTrajectory) -> str:
        """Generate a readable pattern name."""
        steps = trajectory.step_sequence.split(" → ")
        if not steps:
            return "未知路径"
        if steps[0] == "identity_confirm" and len(steps) > 1:
            return f"标准路径 ({len(steps)}步)"
        if "rejection" in steps and "retention" in steps:
            return "挽留成功路径"
        if "rejection" in steps and "retention" not in steps:
            return "拒绝后放弃路径"
        return f"路径 ({len(steps)}步)"

    def _generate_description(self, trajectory: ConversationTrajectory) -> str:
        """Generate a pattern description."""
        steps = trajectory.step_sequence.split(" → ")

        descriptions = {
            "identity_confirm": "身份确认",
            "contract_explain": "合同说明",
            "commitment_request": "请求承诺",
            "commitment_obtain": "获取承诺",
            "rejection": "用户拒绝",
            "retention": "挽留尝试",
            "safety_reminder": "安全提醒",
            "end": "结束",
        }

        desc_list = [descriptions.get(s, s) for s in steps]
        return " → ".join(desc_list)