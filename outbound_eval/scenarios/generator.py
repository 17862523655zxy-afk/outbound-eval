"""Scenario generator for creating test scenarios."""

from typing import Optional
import uuid
from outbound_eval.scenarios.personas import (
    UserPersona,
    PersonaType,
    PERSONA_TEMPLATES,
)
from outbound_eval.dataset.task import EvaluationTask, DifficultyLevel


class ScenarioGenerator:
    """Generates test scenarios for evaluation."""

    def __init__(self):
        """Initialize the generator."""
        self.persona_templates = PERSONA_TEMPLATES

    def generate(
        self,
        task: EvaluationTask,
        num_scenarios: int = 10,
        difficulty_distribution: Optional[dict[str, float]] = None,
    ) -> list[dict]:
        """Generate scenarios for a task.

        Args:
            task: The evaluation task
            num_scenarios: Number of scenarios to generate
            difficulty_distribution: Optional distribution of persona types

        Returns:
            List of scenario dicts
        """
        scenarios = []

        # Default distribution
        if difficulty_distribution is None:
            difficulty_distribution = {
                "cooperative": 0.3,
                "indecisive": 0.2,
                "rejection": 0.25,
                "emotional": 0.15,
                "off_topic": 0.1,
            }

        # Generate scenarios based on distribution
        for i in range(num_scenarios):
            # Select persona type based on distribution
            persona_type = self._select_persona_type(difficulty_distribution)

            # Create persona
            persona = self._create_persona(persona_type, task.variables)

            # Create scenario
            scenario = {
                "scenario_id": f"{task.task_id}_scenario_{i+1:03d}",
                "task_id": task.task_id,
                "persona": persona.model_dump(),
                "difficulty": persona.get_difficulty_label(),
                "event_injections": task.injected_events[:],
            }

            scenarios.append(scenario)

        return scenarios

    def _select_persona_type(
        self, distribution: dict[str, float]
    ) -> PersonaType:
        """Select a persona type based on distribution.

        Args:
            distribution: Distribution dict

        Returns:
            Selected PersonaType
        """
        import random

        # Build cumulative distribution
        types = list(distribution.keys())
        weights = list(distribution.values())

        selected = random.choices(types, weights=weights, k=1)[0]

        # Map string to PersonaType
        type_map = {
            "cooperative": PersonaType.COOPERATIVE,
            "indecisive": PersonaType.INDECISIVE,
            "rejection": PersonaType.REJECTION,
            "emotional": PersonaType.EMOTIONAL,
            "off_topic": PersonaType.OFF_TOPIC,
        }

        return type_map.get(selected, PersonaType.COOPERATIVE)

    def _create_persona(
        self, persona_type: PersonaType, variables: dict
    ) -> UserPersona:
        """Create a persona instance.

        Args:
            persona_type: Type of persona
            variables: Task variables

        Returns:
            UserPersona instance
        """
        # Get template
        template = self.persona_templates.get(
            persona_type, PERSONA_TEMPLATES[PersonaType.COOPERATIVE]
        )

        # Create copy with unique ID
        persona = template.model_copy()
        persona.persona_id = f"{persona_type.value}_{uuid.uuid4().hex[:8]}"

        return persona

    def generate_by_task_difficulty(
        self, task: EvaluationTask
    ) -> list[dict]:
        """Generate scenarios based on task difficulty.

        Args:
            task: The evaluation task

        Returns:
            List of scenarios
        """
        if task.difficulty == DifficultyLevel.EASY:
            # Mostly cooperative
            return self.generate(task, num_scenarios=5, difficulty_distribution={
                "cooperative": 0.7,
                "indecisive": 0.2,
                "rejection": 0.1,
                "emotional": 0.0,
                "off_topic": 0.0,
            })
        elif task.difficulty == DifficultyLevel.MEDIUM:
            # Mixed difficulty
            return self.generate(task, num_scenarios=10, difficulty_distribution={
                "cooperative": 0.3,
                "indecisive": 0.25,
                "rejection": 0.25,
                "emotional": 0.1,
                "off_topic": 0.1,
            })
        else:  # HARD
            # High difficulty
            return self.generate(task, num_scenarios=15, difficulty_distribution={
                "cooperative": 0.1,
                "indecisive": 0.2,
                "rejection": 0.3,
                "emotional": 0.25,
                "off_topic": 0.15,
            })