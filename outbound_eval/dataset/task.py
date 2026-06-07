"""Evaluation task definition."""

from enum import Enum
from typing import ClassVar, Literal
from pydantic import BaseModel, ConfigDict, Field

# Priority levels for success/failure conditions
PriorityLevel = Literal["P0", "P1", "P2", "P3", "P4"]

# Check types for success/failure conditions
CheckType = Literal["rule", "llm", "hybrid"]


class DifficultyLevel(str, Enum):
    """Difficulty levels for evaluation tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SuccessCondition(BaseModel):
    """A single success condition for a task."""

    condition_id: str = Field(description="Unique identifier for this condition")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Description of what this condition checks")
    priority: PriorityLevel = Field(
        default="P1", description="Priority level: P0(mandatory) to P4(optional)"
    )
    check_type: CheckType = Field(
        default="rule", description="Check type: rule | llm | hybrid"
    )
    check_config: dict[str, object] = Field(
        default_factory=dict, description="Configuration for the check"
    )
    weight: float = Field(default=1.0, description="Weight for scoring")


class FailureCondition(BaseModel):
    """A single failure condition - things that must NOT happen."""

    condition_id: str = Field(description="Unique identifier for this condition")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Description of what this condition checks")
    priority: PriorityLevel = Field(
        default="P1", description="Priority level: P0(critical) to P4(minor)"
    )
    check_type: CheckType = Field(
        default="rule", description="Check type: rule | llm | hybrid"
    )
    check_config: dict[str, object] = Field(
        default_factory=dict, description="Configuration for the check"
    )
    penalty_weight: float = Field(default=1.0, description="How much this failure reduces score")


class EvaluationTask(BaseModel):
    """Definition of an evaluation task."""

    task_id: str = Field(description="Unique task identifier")
    name: str = Field(description="Task name")
    description: str = Field(description="Task description")
    skill_name: str = Field(description="Name of the skill script to use")
    variables: dict[str, object] = Field(
        default_factory=dict, description="Variables for the skill script"
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM, description="Task difficulty level"
    )
    difficulty_levels: list[DifficultyLevel] = Field(
        default_factory=lambda: [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
        description="实验切片用的多档难度；空列表则只用 difficulty 单档",
    )
    scenarios_per_level: int = Field(
        default=10, description="每档难度生成的 case 数（实验矩阵用）"
    )

    # Success criteria
    success_criteria: list[SuccessCondition] = Field(
        default_factory=list, description="List of success conditions"
    )
    pass_threshold: float = Field(
        default=0.7, description="Minimum coverage rate to pass (0.0-1.0)"
    )

    # Failure criteria – things that must NOT happen
    failure_criteria: list[FailureCondition] = Field(
        default_factory=list, description="List of failure conditions"
    )

    # Judge weights (optional override)
    judge_weights: dict[str, float] | None = Field(
        default=None, description="Custom judge weights"
    )

    # Expected outcome
    expected_outcome: dict[str, object] = Field(
        default_factory=dict, description="Expected task outcome"
    )

    # Event injection config
    injected_events: list[str] = Field(
        default_factory=list, description="Event IDs to inject during evaluation"
    )

    model_config: ClassVar[ConfigDict] = {
        "json_schema_extra": {
            "example": {
                "task_id": "feimaotui_contract",
                "name": "飞毛腿骑手合同签署通知",
                "description": "通知骑手合同已签署完成并提醒完成配送",
                "skill_name": "feimaotui",
                "difficulty": "medium",
                "success_criteria": [
                    {
                        "condition_id": "identity_confirmed",
                        "name": "身份确认",
                        "check_type": "rule",
                    }
                ],
            }
        }
    }