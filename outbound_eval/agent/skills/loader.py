"""Skill script loader."""

import yaml
from pathlib import Path
from typing import Optional
from outbound_eval.agent.skills.base import SkillScript


class SkillLoader:
    """Loader for skill scripts."""

    def __init__(self, scripts_dir: Optional[str] = None):
        """Initialize the loader.

        Args:
            scripts_dir: Directory containing skill script YAML files.
        """
        if scripts_dir:
            self.scripts_dir = Path(scripts_dir)
        else:
            # /path/to/outbound_eval/outbound_eval/agent/skills/loader.py
            # → /path/to/outbound_eval/outbound_eval/agent/skills/scripts
            package_dir = Path(__file__).parent.parent.parent  # Go up to outbound_eval
            self.scripts_dir = package_dir / "agent" / "skills" / "scripts"

    def load(self, skill_name: str) -> SkillScript:
        """Load a skill script by name.

        Args:
            skill_name: Skill name (filename without .yaml extension)

        Returns:
            The loaded SkillScript
        """
        script_file = self.scripts_dir / f"{skill_name}.yaml"
        if not script_file.exists():
            raise FileNotFoundError(f"Skill script not found: {script_file}")

        with open(script_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return SkillScript(**data)

    def load_all(self) -> dict[str, SkillScript]:
        """Load all skill scripts.

        Returns:
            Dict mapping skill name to SkillScript
        """
        scripts = {}
        if not self.scripts_dir.exists():
            return scripts

        for script_file in self.scripts_dir.glob("*.yaml"):
            try:
                skill = self.load(script_file.stem)
                scripts[skill.name] = skill
            except Exception as e:
                print(f"Warning: Failed to load {script_file}: {e}")

        return scripts