"""Sample test for benchmark runner."""

import pytest
from outbound_eval.dataset.task import EvaluationTask, SuccessCondition, DifficultyLevel


def test_benchmark_runner_init():
    """Test benchmark runner initialization."""
    # Note: Full runner init requires API key, so we just test the class exists
    from outbound_eval.benchmark.runner import BenchmarkRunner
    assert BenchmarkRunner is not None


def test_task_creation():
    """Test task creation."""
    task = EvaluationTask(
        task_id="test_task",
        name="Test Task",
        description="Test description",
        skill_name="test_skill",
        difficulty=DifficultyLevel.EASY,
        success_criteria=[
            SuccessCondition(
                condition_id="test_condition",
                name="Test Condition",
                description="Test condition description",
            )
        ],
    )

    assert task.task_id == "test_task"
    assert task.difficulty == DifficultyLevel.EASY
    assert len(task.success_criteria) == 1