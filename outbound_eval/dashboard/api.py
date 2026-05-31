"""Dashboard API."""

import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional
from outbound_eval.dataset.store import DataStore
from outbound_eval.analytics.persona_metrics import PersonaMetricsAnalyzer
from outbound_eval.analytics.difficulty import DifficultyAnalyzer
from outbound_eval.analytics.trajectory import SuccessPatternAnalyzer
from outbound_eval.visualization.heatmap import FailureHeatmap
from outbound_eval.visualization.failure_tree import FailureTreeGenerator
from outbound_eval.benchmark.monitor import RunMonitor
from outbound_eval.analyzer.failure_analyzer import FailureAnalyzer

app = FastAPI(title="Outbound Agent Evaluation Dashboard")

store = DataStore()
persona_analyzer = PersonaMetricsAnalyzer()
difficulty_analyzer = DifficultyAnalyzer()
trajectory_analyzer = SuccessPatternAnalyzer()
heatmap_generator = FailureHeatmap()
tree_generator = FailureTreeGenerator()


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
def update_settings(payload: dict):
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
def list_results(task_id: Optional[str] = None):
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
                "persona_profile": data.get("persona_profile", {}),
                "conversation_memory": data.get("conversation_memory", {}),
                "state_transitions": data.get("state_transitions", []),
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
def get_success_rate(task_id: Optional[str] = None):
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
def get_persona_metrics(task_id: Optional[str] = None):
    """Get metrics by persona type."""
    result_files = store.list_results(task_id)
    results = [store.load_result(f) for f in result_files]

    if not results:
        return {"persona_metrics": []}

    report = persona_analyzer.analyze(results)
    return report.model_dump()


@app.get("/api/v1/stats/difficulty")
def get_difficulty_metrics(task_id: Optional[str] = None):
    """Get metrics by difficulty level."""
    result_files = store.list_results(task_id)
    results = [store.load_result(f) for f in result_files]

    if not results:
        return {"difficulty_metrics": []}

    report = difficulty_analyzer.analyze(results)
    return report.model_dump()


@app.get("/api/v1/stats/trajectories")
def get_trajectories(task_id: Optional[str] = None):
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
def analyze_failures(task_id: Optional[str] = None):
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
    difficulty: Optional[str] = None,
    persona_type: Optional[str] = None,
    run_name: Optional[str] = None,
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
                "skill_name": t.skill_name,
            }
            for t in tasks
        ]
    }


@app.get("/api/v1/results/{task_id}/{scenario_id}")
def get_result_detail(task_id: str, scenario_id: str):
    """Get detailed result for a single scenario."""
    result = store.load_result_by_scenario(task_id, scenario_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


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
        return HTMLResponse(content=f.read())


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


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()