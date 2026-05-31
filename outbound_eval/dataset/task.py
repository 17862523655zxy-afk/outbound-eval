"""Evaluation task definition."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


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
    check_type: str = Field(
        default="rule", description="Check type: rule, llm, or hybrid"
    )
    check_config: dict[str, Any] = Field(
        default_factory=dict, description="Configuration for the check"
    )
    weight: float = Field(default=1.0, description="Weight for scoring")


class EvaluationTask(BaseModel):
    """Definition of an evaluation task."""

    task_id: str = Field(description="Unique task identifier")
    name: str = Field(description="Task name")
    description: str = Field(description="Task description")
    skill_name: str = Field(description="Name of the skill script to use")
    variables: dict[str, Any] = Field(
        default_factory=dict, description="Variables for the skill script"
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM, description="Task difficulty level"
    )

    # Success criteria
    success_criteria: list[SuccessCondition] = Field(
        default_factory=list, description="List of success conditions"
    )
    pass_threshold: float = Field(
        default=0.7, description="Minimum coverage rate to pass (0.0-1.0)"
    )

    # Expected outcome
    expected_outcome: dict[str, Any] = Field(
        default_factory=dict, description="Expected task outcome"
    )

    # Event injection config
    injected_events: list[str] = Field(
        default_factory=list, description="Event IDs to inject during evaluation"
    )

    model_config = {
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