"""Dashboard API."""

import threading
from typing import Any, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from outbound_eval.dataset.store import DataStore
from outbound_eval.dataset.loader import TaskLoader
from outbound_eval.analytics.persona_metrics import PersonaMetricsAnalyzer
from outbound_eval.analytics.difficulty import DifficultyAnalyzer
from outbound_eval.analytics.trajectory import SuccessPatternAnalyzer
from outbound_eval.visualization.heatmap import FailureHeatmap
from outbound_eval.visualization.failure_tree import FailureTreeGenerator
from outbound_eval.benchmark.monitor import RunMonitor
from outbound_eval.analyzer.failure_analyzer import FailureAnalyzer
from outbound_eval.dashboard.assets import AssetRegistry

app = FastAPI(title="Outbound Agent Evaluation Dashboard")

store = DataStore()
persona_analyzer = PersonaMetricsAnalyzer()
difficulty_analyzer = DifficultyAnalyzer()
trajectory_analyzer = SuccessPatternAnalyzer()
heatmap_generator = FailureHeatmap()
tree_generator = FailureTreeGenerator()
assets = AssetRegistry()


@app.get("/api/v1/settings")
def get_settings():
    """Get current application settings (sensitive fields masked)."""
    from outbound_eval.infra.config import settings
    return {
        "openai_api_key": "***" if settings.openai_api_key else "",
        "llm_model": settings.llm_model,
        "judge_llm_model": settings.judge_llm_model,
        "base_url": settings.base_url,
        "token_price_per_1k": settings.token_price_per_1k,
        "max_turns_per_call": settings.max_turns_per_call,
        "target_turns": settings.target_turns,
        "target_cost_per_success": settings.target_cost_per_success,
        "dashboard_host": settings.dashboard_host,
        "dashboard_port": settings.dashboard_port,
        "log_level": settings.log_level,
    }


@app.put("/api/v1/settings")
def update_settings(payload: dict[str, Any]):
    """Update application settings and write to .env file."""
    from outbound_eval.infra.config import Settings

    env_path = Path(".env")
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    # Build key -> line index mapping
    key_map = {}
    for i, line in enumerate(lines):
        if "=" in line and not line.strip().startswith("#"):
            k = line.split("=", 1)[0].strip()
            key_map[k] = i

    updates = {
        "OPENAI_API_KEY": payload.get("openai_api_key", ""),
        "LLM_MODEL": payload.get("llm_model", "deepseek-chat"),
        "JUDGE_LLM_MODEL": payload.get("judge_llm_model", "deepseek-chat"),
        "BASE_URL": payload.get("base_url", "https://api.deepseek.com"),
        "TOKEN_PRICE_PER_1K": str(payload.get("token_price_per_1k", 0.0001)),
        "MAX_TURNS_PER_CALL": str(payload.get("max_turns_per_call", 20)),
        "TARGET_TURNS": str(payload.get("target_turns", 8.0)),
        "TARGET_COST_PER_SUCCESS": str(payload.get("target_cost_per_success", 0.1)),
        "DASHBOARD_HOST": payload.get("dashboard_host", "0.0.0.0"),
        "DASHBOARD_PORT": str(payload.get("dashboard_port", 8000)),
        "LOG_LEVEL": payload.get("log_level", "INFO"),
    }

    # Skip empty API key to avoid overwriting with blank
    if not updates["OPENAI_API_KEY"]:
        updates.pop("OPENAI_API_KEY")

    for k, v in updates.items():
        line = f'{k}={v}'
        if k in key_map:
            lines[key_map[k]] = line
        else:
            lines.append(line)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Reload settings
    new_settings = Settings()
    return {"status": "saved", "settings": {
        "openai_api_key": "***" if new_settings.openai_api_key else "",
        "llm_model": new_settings.llm_model,
        "judge_llm_model": new_settings.judge_llm_model,
        "base_url": new_settings.base_url,
        "token_price_per_1k": new_settings.token_price_per_1k,
        "max_turns_per_call": new_settings.max_turns_per_call,
        "target_turns": new_settings.target_turns,
        "target_cost_per_success": new_settings.target_cost_per_success,
        "dashboard_host": new_settings.dashboard_host,
        "dashboard_port": new_settings.dashboard_port,
        "log_level": new_settings.log_level,
    }}


@app.get("/api/v1/results")
def list_results(task_id: str | None = None):
    """List evaluation results with full data."""
    result_files = store.list_results(task_id)
    summaries = []
    for f in result_files:
        try:
            data = store.load_result(f)
            summaries.append({
                "run_id": data.get("run_id", ""),
                "run_name": data.get("run_name", ""),
                "task_id": data.get("task_id", ""),
                "scenario_id": data.get("scenario_id", ""),
                "persona_type": data.get("persona_type", ""),
                "difficulty": data.get("difficulty", ""),
                "overall_score": data.get("overall_score", 0),
                "passed": data.get("passed", False),
                "pass_threshold": data.get("pass_threshold", 0.7),
                "timestamp": f.stat().st_mtime,
                "task_success": data.get("task_success", 0),
                "flow_adherence": data.get("flow_adherence", 0),
                "flow_adherence_detail": data.get("flow_adherence_detail"),
                "state_tracking": data.get("state_tracking", 0),
                "state_tracking_detail": data.get("state_tracking_detail"),
                "compliance": data.get("compliance", 0),
                "naturalness": data.get("naturalness", 0),
                "recovery": data.get("recovery", 0),
                "efficiency": data.get("efficiency", 0),
                "efficiency_detail": data.get("efficiency_detail"),
                "dialogue_history": data.get("dialogue_history", []),
                "failure_reasons": data.get("failure_reasons", []),
                "improvement_suggestions": data.get("improvement_suggestions", []),
                "criterion_details": data.get("criterion_details", []),
                "gold_comparison": data.get("gold_comparison"),
                "persona_profile": data.get("persona_profile", {}),
                "conversation_memory": data.get("conversation_memory", {}),
                "state_transitions": data.get("state_transitions", []),
                "state_path": data.get("state_path", []),
                "triggered_events": data.get("triggered_events", []),
                "agent_state": data.get("agent_state", {}),
                "total_tokens": data.get("total_tokens", 0),
                "cost": data.get("cost", 0),
                "elapsed_seconds": data.get("elapsed_seconds", 0),
            })
        except Exception:
            pass
    return {"results": summaries}


@app.get("/api/v1/stats/success_rate")
def get_success_rate(task_id: str | None = None):
    """Get success rate statistics."""
    result_files = store.list_results(task_id)
    results = [store.load_result(f) for f in result_files]

    if not results:
        return {"success_rate": 0.0, "total": 0}

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))

    return {
        "success_rate": (passed / total * 100) if total > 0 else 0,
        "total": total,
        "passed": passed,
        "failed": total - passed,
    }


@app.get("/api/v1/stats/persona_metrics")
def get_persona_metrics(task_id: str | None = None):
    """Get metrics by persona type."""
    result_files = store.list_results(task_id)
    results = [store.load_result(f) for f in result_files]

    if not results:
        return {"persona_metrics": []}

    report = persona_analyzer.analyze(results)
    return report.model_dump()


@app.get("/api/v1/stats/difficulty")
def get_difficulty_metrics(task_id: str | None = None):
    """Get metrics by difficulty level."""
    result_files = store.list_results(task_id)
    results = [store.load_result(f) for f in result_files]

    if not results:
        return {"difficulty_metrics": []}

    report = difficulty_analyzer.analyze(results)
    return report.model_dump()


@app.get("/api/v1/stats/trajectories")
def get_trajectories(task_id: str | None = None):
    """Get success trajectory patterns."""
    result_files = store.list_results(task_id)
    results = [store.load_result(f) for f in result_files]

    if not results:
        return {"top_patterns": []}

    report = trajectory_analyzer.analyze(results)
    return report.model_dump()


@app.get("/api/v1/stats/heatmap")
def get_heatmap(type: str = "scenario"):
    """Get failure heatmap."""
    result_files = store.list_results()
    results = [store.load_result(f) for f in result_files]

    if not results:
        return {"cells": []}

    if type == "scenario":
        return heatmap_generator.generate_by_scenario_metric(results).model_dump()
    else:
        return heatmap_generator.generate_by_persona_metric(results).model_dump()


@app.post("/api/v1/analyze/failures")
def analyze_failures(task_id: str | None = None):
    """Analyze failures using FailureAnalyzer."""
    result_files = store.list_results(task_id)
    results = [store.load_result(f) for f in result_files]

    if not results:
        return {"total_failures": 0, "failure_distribution": {}, "analyses": []}

    analyzer = FailureAnalyzer()
    all_analyses = []

    for result in results:
        if not result.get("passed", False):
            analyses = analyzer.analyze(result, result.get("dialogue_history", []))
            all_analyses.extend(analyses)

    aggregated = analyzer.aggregate_failures(all_analyses)

    return {
        "total_failures": aggregated["total_failures"],
        "failure_distribution": {k.value: v for k, v in aggregated["failure_distribution"].items()},
        "most_common_failure": aggregated["most_common_failure"].value if aggregated["most_common_failure"] else None,
        "analyses": [a.model_dump() for a in all_analyses[:20]],  # Return first 20 for detail view
    }


# Difficulty-based persona distributions (same as ScenarioGenerator.generate_by_task_difficulty)
_DIFFICULTY_DISTRIBUTIONS = {
    "easy": {"cooperative": 0.7, "indecisive": 0.2, "rejection": 0.1, "emotional": 0.0, "off_topic": 0.0},
    "medium": {"cooperative": 0.3, "indecisive": 0.25, "rejection": 0.25, "emotional": 0.1, "off_topic": 0.1},
    "hard": {"cooperative": 0.1, "indecisive": 0.2, "rejection": 0.3, "emotional": 0.25, "off_topic": 0.15},
}


@app.post("/api/v1/run")
def start_run(
    task_id: str,
    scenarios: int = 5,
    difficulty: str | None = None,
    persona_type: str | None = None,
    run_name: str | None = None,
):
    """Start evaluation run in background with optional difficulty/persona override."""
    from outbound_eval.dataset.loader import TaskLoader
    from outbound_eval.scenarios.generator import ScenarioGenerator
    from outbound_eval.benchmark.pipeline import EvalPipeline
    from outbound_eval.dataset.task import DifficultyLevel

    def run_evaluation():
        try:
            loader = TaskLoader()
            task = loader.load(task_id)

            # Override difficulty if specified
            if difficulty:
                task.difficulty = DifficultyLevel(difficulty)

            generator = ScenarioGenerator()

            # Build distribution
            if persona_type and persona_type != "all":
                # Single persona type: 100% focused
                distribution = {
                    "cooperative": 0.0,
                    "indecisive": 0.0,
                    "rejection": 0.0,
                    "emotional": 0.0,
                    "off_topic": 0.0,
                }
                distribution[persona_type] = 1.0
            elif difficulty and difficulty in _DIFFICULTY_DISTRIBUTIONS:
                distribution = _DIFFICULTY_DISTRIBUTIONS[difficulty]
            else:
                # Fallback to task's own difficulty distribution
                distribution = None

            scenario_list = generator.generate(
                task, num_scenarios=scenarios, difficulty_distribution=distribution
            )

            pipeline = EvalPipeline()
            pipeline.run(task, scenario_list, run_name=run_name)
        except Exception as e:
            monitor = RunMonitor()
            monitor.fail_run(str(e))

    # Start in background thread
    thread = threading.Thread(target=run_evaluation, daemon=True)
    thread.start()

    return {
        "status": "started",
        "task_id": task_id,
        "scenarios": scenarios,
        "difficulty": difficulty,
        "persona_type": persona_type,
        "run_name": run_name,
    }


@app.get("/api/v1/run-status")
def get_run_status():
    """Get current run status for real-time monitoring."""
    monitor = RunMonitor()
    return monitor.state.to_dict()


@app.get("/api/v1/tasks")
def list_tasks():
    """List available evaluation tasks."""
    from outbound_eval.dataset.loader import TaskLoader
    loader = TaskLoader()
    tasks = loader.load_all()
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "description": t.description,
                "difficulty": t.difficulty.value,
                "difficulty_levels": [lvl.value for lvl in t.difficulty_levels],
                "scenarios_per_level": t.scenarios_per_level,
                "skill_name": t.skill_name,
            }
            for t in tasks
        ]
    }


@app.get("/api/v1/benchmarks")
def list_benchmarks():
    """List all benchmark tasks with summary info."""
    loader = TaskLoader()
    tasks = loader.load_all()
    return {
        "benchmarks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "description": t.description,
                "difficulty": t.difficulty.value,
                "difficulty_levels": [lvl.value for lvl in t.difficulty_levels],
                "scenarios_per_level": t.scenarios_per_level,
                "skill_name": t.skill_name,
                "case_count": len(t.success_criteria) + len(t.failure_criteria),
                "success_count": len(t.success_criteria),
                "failure_count": len(t.failure_criteria),
            }
            for t in tasks
        ]
    }


@app.get("/api/v1/benchmarks/{task_id}")
def get_benchmark(task_id: str):
    """Get full benchmark details as structured JSON."""
    detail = assets.get_task(task_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return detail


@app.put("/api/v1/benchmarks/{task_id}")
def update_benchmark(task_id: str, payload: dict[str, Any]):
    """Update an existing benchmark task. Payload contains editable fields only.

    Editable: ``name``, ``description``, ``difficulty``, ``pass_threshold``,
    ``judge_weights``, ``success_criteria``, ``failure_criteria``,
    ``injected_events``, ``expected_outcome``.
    Protected: ``task_id``, ``skill_name``, ``variables``.
    """
    import traceback
    import yaml
    from outbound_eval.dataset.task import EvaluationTask, DifficultyLevel
    from outbound_eval.dataset.loader import TaskLoader

    task_file = assets.tasks_dir / f"{task_id}.yaml"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    try:
        # Use TaskLoader (canonical, returns full Pydantic-valid data)
        current = TaskLoader().load(task_id)
        merged = current.model_dump(mode="json")

        editable_keys = (
            "name", "description", "difficulty", "pass_threshold",
            "judge_weights", "success_criteria", "failure_criteria",
            "injected_events", "expected_outcome",
        )
        for k in editable_keys:
            if k in payload and payload[k] is not None:
                merged[k] = payload[k]

        # Re-validate (catches bad difficulty, missing required fields, etc.)
        new_task = EvaluationTask(**merged)
        new_task.task_id = task_id
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"校验失败: {str(e)[:300]}")

    try:
        task_file.write_text(
            yaml.safe_dump(
                new_task.model_dump(mode="json"),
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"写入失败: {e}")

    return {"task_id": task_id, "status": "updated", "path": str(task_file)}


@app.delete("/api/v1/benchmarks/{task_id}")
def delete_benchmark(task_id: str):
    """Delete a benchmark task YAML file from disk."""
    task_file = assets.tasks_dir / f"{task_id}.yaml"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    try:
        task_file.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
    return {"task_id": task_id, "status": "deleted"}


@app.get("/api/v1/assets/summary")
def get_assets_summary():
    """Aggregate catalog of all evaluation assets (tasks, gold, personas)."""
    return {
        "tasks": assets.list_tasks(),
        "gold_conversations": assets.list_gold(),
        "personas": assets.list_personas(),
    }


@app.post("/api/v1/instructions/parse")
def parse_instruction_endpoint(payload: dict[str, Any]):
    """Parse a natural-language task instruction and return a draft task.

    Body: ``{"text": str, "task_id": str, "save": bool,
              "excel_path": Optional[str]}``.

    When ``save`` is true, the generated YAML is written to
    ``data/benchmarks/tasks/<task_id>.yaml`` and the resolved path is
    returned. When ``excel_path`` is provided, the .xlsx file is read
    directly (overrides ``text`` if both are present).
    """
    import traceback
    from pathlib import Path
    from outbound_eval.dataset.instruction_parser import (
        parse_and_save, ParsedInstruction, to_evaluation_task, parse_excel_file,
    )

    task_id = (payload.get("task_id") or "").strip()
    do_save = bool(payload.get("save", False))
    excel_path = (payload.get("excel_path") or "").strip()
    text = (payload.get("text") or "").strip()

    if not task_id:
        raise HTTPException(status_code=400, detail="task_id 不能为空")

    try:
        if excel_path:
            try:
                out_path = None
                if do_save:
                    tasks_dir = Path(__file__).resolve().parent.parent.parent / "data" / "benchmarks" / "tasks"
                    tasks_dir.mkdir(parents=True, exist_ok=True)
                    out_path = tasks_dir / f"{task_id}.yaml"
                parsed, task, saved, raw_text = parse_excel_file(Path(excel_path), task_id, out_path)
            except RuntimeError as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            if not text:
                raise HTTPException(status_code=400, detail="text 或 excel_path 至少需要提供一个")
            if do_save:
                tasks_dir = Path(__file__).resolve().parent.parent.parent / "data" / "benchmarks" / "tasks"
                tasks_dir.mkdir(parents=True, exist_ok=True)
                out_path = tasks_dir / f"{task_id}.yaml"
                parsed, task, saved = parse_and_save(text, task_id, out_path)
                raw_text = text
            else:
                from outbound_eval.dataset.instruction_parser import parse_instruction
                parsed = parse_instruction(text)
                task = to_evaluation_task(parsed, task_id=task_id)
                saved = None
                raw_text = text

        return {
            "task_id": task_id,
            "parsed": parsed.model_dump(),
            "task_yaml": task.model_dump(mode="json"),
            "saved_path": str(saved) if saved else None,
            "errors": parsed.errors,
            "raw_text_preview": raw_text[:200],
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"解析失败: {e}")


@app.get("/api/v1/assets/gold/{conversation_id}")
def get_gold_detail(conversation_id: str):
    """Get a single gold conversation's full content."""
    detail = assets.get_gold(conversation_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Gold conversation not found: {conversation_id}")
    return detail


@app.get("/api/v1/assets/personas/{persona_type}")
def get_persona_detail(persona_type: str):
    """Get a single persona template's full content."""
    detail = assets.get_persona(persona_type)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Persona not found: {persona_type}")
    return detail


@app.get("/api/v1/results/{task_id}/report")
def get_result_report(task_id: str):
    """Generate evaluation report JSON for download."""
    from datetime import datetime
    result_files = store.list_results(task_id)
    results = [store.load_result(f) for f in result_files]

    if not results:
        return {"error": "No results found", "results": []}

    # Group by run
    runs = {}
    for r in results:
        rid = r.get("run_id", "unknown")
        if rid not in runs:
            runs[rid] = {"run_id": rid, "run_name": r.get("run_name", rid), "timestamp": r.get("timestamp", 0), "cases": []}
        runs[rid]["cases"].append(r)

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))

    return {
        "report_time": datetime.now().isoformat(),
        "task_id": task_id,
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "avg_score": round(sum(r.get("overall_score", 0) for r in results) / total, 1) if total > 0 else 0,
        },
        "runs": [
            {
                "run_id": rid,
                "run_name": rd["run_name"],
                "timestamp": rd["timestamp"],
                "cases": [
                    {
                        "scenario_id": c.get("scenario_id", ""),
                        "persona_type": c.get("persona_type", ""),
                        "difficulty": c.get("difficulty", ""),
                        "passed": c.get("passed", False),
                        "overall_score": c.get("overall_score", 0),
                        "metrics": {
                            "task_success": c.get("task_success", 0),
                            "flow_adherence": c.get("flow_adherence", 0),
                            "compliance": c.get("compliance", 0),
                            "recovery": c.get("recovery", 0),
                            "naturalness": c.get("naturalness", 0),
                        },
                        "failure_reasons": c.get("failure_reasons", []),
                        "improvement_suggestions": c.get("improvement_suggestions", []),
                    }
                    for c in rd["cases"]
                ],
            }
            for rid, rd in runs.items()
        ],
    }


@app.get("/api/v1/results/{task_id}/{scenario_id}/dialogue")
def get_dialogue(task_id: str, scenario_id: str):
    """Get dialogue history for a scenario."""
    result = store.load_result_by_scenario(task_id, scenario_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return {
        "dialogue_history": result.get("dialogue_history", []),
        "agent_state": result.get("agent_state", {}),
    }


@app.get("/")
def dashboard():
    """Dashboard HTML page."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f:
        html_text = f.read()
    return HTMLResponse(
        content=html_text,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/v1/results/{task_id}/{scenario_id}")
def get_result_detail(task_id: str, scenario_id: str):
    """Get detailed result for a single scenario."""
    result = store.load_result_by_scenario(task_id, scenario_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


# ---------------------------------------------------------------------------
# HTML / Markdown report endpoints
# ---------------------------------------------------------------------------

_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "reports"


def _list_report_files(task_id: str | None = None) -> list[Path]:
    if not _REPORTS_DIR.exists():
        return []
    files = list(_REPORTS_DIR.glob("*.html"))
    if task_id:
        # Files named ``<run_id>_<timestamp>.html``; run_id starts with ``run_<date>``
        # so we cannot easily filter by task_id — just return all, sorted.
        pass
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


@app.get("/api/v1/reports/list")
def list_reports(task_id: str | None = None):
    """List available HTML reports on disk."""
    items = []
    for f in _list_report_files(task_id):
        items.append({
            "filename": f.name,
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "mtime": f.stat().st_mtime,
            "download_url": f"/api/v1/reports/download/{f.name}",
        })
    return {"reports": items}


@app.get("/api/v1/reports/download/{filename}")
def download_report(filename: str):
    """Stream a generated HTML report."""
    from fastapi.responses import FileResponse
    target = _REPORTS_DIR / filename
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Report not found: {filename}")
    return FileResponse(target, media_type="text/html", filename=filename)


@app.post("/api/v1/reports/generate")
def trigger_report_generation(payload: dict[str, Any]):
    """Generate a fresh report from raw result files for a task.

    Body: ``{"task_id": str, "run_id": Optional[str], "format": "html"|"md"|"pdf"}``.
    When ``run_id`` is provided only that run is included; otherwise all
    runs for the task are.
    """
    import traceback
    from datetime import datetime
    from outbound_eval.reporting.report import ReportBuilder
    from outbound_eval.reporting.pdf import PdfRenderError
    from outbound_eval.dataset.loader import TaskLoader

    task_id = (payload.get("task_id") or "").strip()
    run_id = (payload.get("run_id") or "").strip() or None
    fmt = (payload.get("format") or "html").lower()
    if fmt not in ("html", "md", "pdf"):
        raise HTTPException(status_code=400, detail=f"不支持的 format: {fmt!r}")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id 不能为空")

    try:
        loader = TaskLoader()
        try:
            task = loader.load(task_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        result_files = store.list_results(task_id)
        results = [store.load_result(f) for f in result_files]
        if run_id:
            results = [r for r in results if r.get("run_id") == run_id]
        if not results:
            raise HTTPException(status_code=404, detail="该 task 暂无 result")

        effective_run_id = run_id or results[0].get("run_id", "manual")
        rb = ReportBuilder(task, results, run_id=effective_run_id)
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = {"html": "html", "md": "md", "pdf": "pdf"}[fmt]
        out_path = _REPORTS_DIR / f"{effective_run_id}_{ts}.{ext}"

        try:
            rb.save(out_path, format=fmt)
        except PdfRenderError as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "status": "generated",
            "task_id": task_id,
            "run_id": effective_run_id,
            "format": fmt,
            "path": str(out_path),
            "download_url": f"/api/v1/reports/download/{out_path.name}",
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成失败: {e}")


@app.post("/api/v1/experiments/run")
def run_experiment(
    baseline_task_id: str,
    candidate_task_id: str,
    scenarios: int = 5,
    difficulty: str | None = None,
    persona_type: str | None = None,
    difficulty_levels: list[str] | None = None,
    scenarios_per_level: int | None = None,
):
    """Run A/B experiment comparing two task configurations.

    New (v2) parameters (preferred):
      - ``difficulty_levels``: list of difficulty levels to slice on
        (e.g. ``["easy", "medium", "hard"]``). Falls back to
        ``[difficulty]`` if only one is wanted, or to the
        task's own ``difficulty_levels`` if both are missing.
      - ``scenarios_per_level``: how many cases per level per side.
        Falls back to ``scenarios`` if not set.

    Legacy parameters are still honored:
      - ``difficulty`` (single level), ``scenarios`` (total count).
    """
    from outbound_eval.experiments.runner import ExperimentRunner, ExperimentConfig
    from outbound_eval.experiments.comparison import ExperimentComparisonAnalyzer
    from outbound_eval.dataset.loader import TaskLoader

    # Resolve difficulty levels
    if difficulty_levels:
        levels = difficulty_levels
    elif difficulty:
        levels = [difficulty]
    else:
        # Try the task's own difficulty_levels; fall back to all three
        try:
            t = TaskLoader().load(baseline_task_id)
            levels = [l.value for l in t.difficulty_levels] or ["easy", "medium", "hard"]
        except Exception:
            levels = ["easy", "medium", "hard"]

    # Resolve per-level scenario count
    spl = scenarios_per_level or scenarios or 5

    config = ExperimentConfig(
        name=f"AB_{baseline_task_id}_vs_{candidate_task_id}",
        baseline_task_id=baseline_task_id,
        candidate_task_id=candidate_task_id,
        difficulty_levels=levels,
        scenarios_per_level=spl,
    )

    def run_ab():
        try:
            runner = ExperimentRunner()
            # eval_pipeline now receives (agent, task_id, level)
            baseline_run, candidate_run = runner.run_experiment(
                config,
                agent_factory=lambda v: None,  # pipeline builds agent internally
                eval_pipeline=lambda agent, task_id, level: _run_eval_for_level(
                    task_id, level, spl, persona_type
                ),
            )

            analyzer = ExperimentComparisonAnalyzer()
            is_cross = baseline_task_id != candidate_task_id
            comparison = analyzer.analyze(
                baseline_run, candidate_run, config.experiment_id,
                is_cross_business=is_cross,
                baseline_task_id=baseline_task_id,
                candidate_task_id=candidate_task_id,
            )

            # Save results (back-compat: still three files, with new fields nested)
            store.save_experiment(config.experiment_id, "baseline", [baseline_run.model_dump()])
            store.save_experiment(config.experiment_id, "candidate", [candidate_run.model_dump()])
            store.save_experiment(config.experiment_id, "comparison", [comparison.model_dump()])
        except Exception as e:
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=run_ab, daemon=True)
    thread.start()

    return {
        "status": "started",
        "experiment_id": config.experiment_id,
        "baseline_version": baseline_task_id,
        "candidate_version": candidate_task_id,
        "difficulty_levels": levels,
        "scenarios_per_level": spl,
        "is_cross_business": baseline_task_id != candidate_task_id,
    }


def _run_eval_for_level(
    task_id: str,
    level: str,
    scenarios: int,
    persona_type: str | None = None,
) -> dict[str, Any]:
    """Helper: run evaluation for a single task at a specific difficulty level.

    Returns a dict with summary fields plus a ``difficulty`` tag, so that
    :class:`ExperimentComparisonAnalyzer` can split by level.
    """
    from outbound_eval.dataset.loader import TaskLoader
    from outbound_eval.scenarios.generator import ScenarioGenerator
    from outbound_eval.benchmark.pipeline import EvalPipeline
    from outbound_eval.dataset.task import DifficultyLevel

    loader = TaskLoader()
    task = loader.load(task_id)
    task.difficulty = DifficultyLevel(level)

    generator = ScenarioGenerator()
    if persona_type and persona_type != "all":
        distribution = {k: 0.0 for k in ["cooperative", "indecisive", "rejection", "emotional", "off_topic"]}
        distribution[persona_type] = 1.0
    else:
        # Lock to a single level's distribution
        distribution = _DIFFICULTY_DISTRIBUTIONS.get(level, None)

    scenario_list = generator.generate(
        task, num_scenarios=scenarios, difficulty_distribution=distribution
    )
    pipeline = EvalPipeline()
    results = pipeline.run(task, scenario_list)

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    scores = [r.get("overall_score", 0.0) for r in results]
    costs = [r.get("cost", 0.0) for r in results]

    return {
        "passed": passed,
        "total": total,
        "success_rate": (passed / total * 100) if total > 0 else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "avg_cost": sum(costs) / len(costs) if costs else 0.0,
        "difficulty": level,
        "results": results,
    }


# Back-compat alias (older callers)
def _run_eval_for_task(
    task_id: str,
    scenarios: int,
    difficulty: Optional[str] = None,
    persona_type: Optional[str] = None,
) -> dict[str, Any]:
    return _run_eval_for_level(
        task_id, difficulty or "medium", scenarios, persona_type
    )


@app.get("/api/v1/experiments")
def list_experiments():
    """List all A/B experiments."""
    experiments = store.list_experiments()
    return {"experiments": experiments}


@app.get("/api/v1/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    """Get A/B experiment details with comparison results."""
    versions = store.load_experiment_versions(experiment_id)
    if not versions:
        raise HTTPException(status_code=404, detail="Experiment not found")

    baseline = versions.get("baseline", {})
    candidate = versions.get("candidate", {})
    comparison = versions.get("comparison", {})

    return {
        "experiment_id": experiment_id,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()