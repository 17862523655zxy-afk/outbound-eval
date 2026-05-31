"""Skill script definition."""

from typing import Any
from pydantic import BaseModel, Field


class SkillScript(BaseModel):
    """Definition of a skill script for an outbound agent."""

    name: str = Field(description="Skill name")
    description: str = Field(description="Skill description")

    # Role and task
    role: str = Field(description="Agent role description")
    task: str = Field(description="Task description")

    # Opening line
    opening_line: str = Field(description="Opening line template")

    # Variables
    variables: list[dict[str, Any]] = Field(
        default_factory=list, description="Variable definitions"
    )

    # Conversation flow
    flow: list[dict[str, Any]] = Field(
        default_factory=list, description="Conversation flow steps"
    )

    # FAQ
    faq: list[dict[str, str]] = Field(
        default_factory=list, description="Frequently asked questions"
    )

    # Constraints
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Constraint rules"
    )

    def fill_variables(self, **kwargs) -> "SkillScript":
        """Create a copy with variables filled.

        Args:
            **kwargs: Variable values

        Returns:
            New SkillScript with filled variables
        """
        import copy
        new_script = copy.deepcopy(self)

        for var in new_script.variables:
            var_name = var.get("name", "")
            if var_name in kwargs:
                value = kwargs[var_name]
                # Replace ${var_name} in all text fields
                new_script.opening_line = new_script.opening_line.replace(
                    f"${{{var_name}}}", str(value)
                )
                new_script.role = new_script.role.replace(f"${{{var_name}}}", str(value))
                new_script.task = new_script.task.replace(f"${{{var_name}}}", str(value))

        return new_script