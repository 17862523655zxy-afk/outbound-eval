"""Asset registry for evaluation data (tasks, gold conversations, personas).

Scans YAML files and Pydantic registries to expose a unified catalog
for the dashboard's read-only views.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from outbound_eval.scenarios.personas import PERSONA_TEMPLATES, UserPersona


def _project_root() -> Path:
    """Return the workspace root containing the `data/` directory."""
    return Path(__file__).resolve().parent.parent.parent


class AssetRegistry:
    """Catalog of evaluation assets on disk.

    Three asset classes are supported:
      * Tasks        — `data/benchmarks/tasks/*.yaml`
      * Gold         — `data/gold/library/**/*.yaml`
      * Personas     — `PERSONA_TEMPLATES` in `scenarios/personas.py`
    """

    def __init__(
        self,
        tasks_dir: Optional[str] = None,
        gold_dir: Optional[str] = None,
    ) -> None:
        root = _project_root()
        self.tasks_dir = Path(tasks_dir) if tasks_dir else root / "data" / "benchmarks" / "tasks"
        self.gold_dir = Path(gold_dir) if gold_dir else root / "data" / "gold" / "library"
        self.results_dir = root / "data" / "results" / "raw"

    # ---------- Tasks ----------

    def list_tasks(self) -> list[dict[str, Any]]:
        """Return a list of evaluation tasks as flat dicts."""
        items: list[dict[str, Any]] = []
        if not self.tasks_dir.exists():
            return items

        for task_file in sorted(self.tasks_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(task_file.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not isinstance(data, dict):
                continue

            task_id = data.get("task_id") or task_file.stem
            success = data.get("success_criteria") or []
            failure = data.get("failure_criteria") or []
            case_count = len(success) + len(failure) if isinstance(success, list) and isinstance(failure, list) else 0

            items.append({
                "task_id": task_id,
                "name": data.get("name", task_id),
                "description": (data.get("description") or "").strip(),
                "difficulty": data.get("difficulty", "medium"),
                "case_count": case_count,
                "success_count": len(success) if isinstance(success, list) else 0,
                "failure_count": len(failure) if isinstance(failure, list) else 0,
                "last_run_at": self._latest_result_mtime(task_id),
            })

        items.sort(key=lambda x: x["task_id"])
        return items

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """Return a full task dict (suitable for detail view) with business-friendly fields."""
        task_file = self.tasks_dir / f"{task_id}.yaml"
        if not task_file.exists():
            return None
        try:
            data = yaml.safe_load(task_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        success = data.get("success_criteria") or []
        failure = data.get("failure_criteria") or []

        return {
            "task_id": data.get("task_id", task_id),
            "name": data.get("name", task_id),
            "description": (data.get("description") or "").strip(),
            "difficulty": data.get("difficulty", "medium"),
            "difficulty_levels": data.get("difficulty_levels") or ["easy", "medium", "hard"],
            "scenarios_per_level": data.get("scenarios_per_level", 10),
            "skill_name": data.get("skill_name", ""),
            "pass_threshold": data.get("pass_threshold", 0.7),
            "judge_weights": data.get("judge_weights") or {},
            "success_criteria": [_condition_to_business(c) for c in success if isinstance(c, dict)],
            "failure_criteria": [_condition_to_business(c) for c in failure if isinstance(c, dict)],
            "last_run_at": self._latest_result_mtime(data.get("task_id", task_id)),
        }

    def _latest_result_mtime(self, task_id: str) -> Optional[float]:
        if not self.results_dir.exists():
            return None
        latest: Optional[float] = None
        for f in self.results_dir.glob(f"{task_id}_*.json"):
            mt = f.stat().st_mtime
            if latest is None or mt > latest:
                latest = mt
        return latest

    # ---------- Gold conversations ----------

    def list_gold(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if not self.gold_dir.exists():
            return items

        for gold_file in sorted(self.gold_dir.rglob("*.yaml")):
            try:
                data = yaml.safe_load(gold_file.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            turns = data.get("turns") or []
            mt = gold_file.stat().st_mtime
            items.append({
                "conversation_id": data.get("conversation_id", gold_file.stem),
                "task_id": data.get("task_id", ""),
                "persona_type": data.get("persona_type", ""),
                "name": data.get("scenario", data.get("conversation_id", gold_file.stem)),
                "quality_level": data.get("quality_level", ""),
                "turn_count": len(turns) if isinstance(turns, list) else 0,
                "updated_at": mt,
            })

        items.sort(key=lambda x: x["conversation_id"])
        return items

    def get_gold(self, conversation_id: str) -> Optional[dict[str, Any]]:
        for gold_file in self.gold_dir.rglob("*.yaml"):
            try:
                data = yaml.safe_load(gold_file.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("conversation_id") == conversation_id or gold_file.stem == conversation_id:
                turns = data.get("turns") or []
                return {
                    "conversation_id": data.get("conversation_id", gold_file.stem),
                    "task_id": data.get("task_id", ""),
                    "persona_type": data.get("persona_type", ""),
                    "name": data.get("scenario", conversation_id),
                    "quality_level": data.get("quality_level", ""),
                    "turns": turns,
                    "tags": data.get("tags", []),
                    "strategy_markers": data.get("strategy_markers", []),
                }
        return None

    # ---------- Personas ----------

    def list_personas(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for ptype, persona in PERSONA_TEMPLATES.items():
            items.append({
                "persona_type": ptype.value,
                "name": persona.name,
            })
        items.sort(key=lambda x: x["persona_type"])
        return items

    def get_persona(self, persona_type: str) -> Optional[dict[str, Any]]:
        persona: Optional[UserPersona] = None
        for ptype, p in PERSONA_TEMPLATES.items():
            if ptype.value == persona_type:
                persona = p
                break
        if persona is None:
            return None

        return {
            "persona_type": persona.persona_type.value,
            "name": persona.name,
            # Enhanced fields (业务化展示)
            "motivation": persona.motivation,
            "current_status": persona.current_status,
            "concerns": persona.concerns,
            "goals": persona.goals,
            "emotional_baseline": persona.emotional_baseline,
            "patience_level": persona.patience_level,
            "speaking_habits": persona.speaking_habits,
            "difficulty_label": persona.get_difficulty_label(),
        }


def _condition_to_business(cond: dict[str, Any]) -> dict[str, Any]:
    """Map a raw success/failure condition into business-facing fields only.

    Preserves ``condition_id`` so the PUT round-trip can rebuild the
    full Pydantic model; the dashboard simply chooses not to display it.
    """
    return {
        "condition_id": cond.get("condition_id", ""),
        "name": cond.get("name", cond.get("condition_id", "")),
        "description": cond.get("description", ""),
        "priority": cond.get("priority", "P1"),
    }
