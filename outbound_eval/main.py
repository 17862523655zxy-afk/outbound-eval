"""CLI main entry point."""

import click
from outbound_eval.dataset.loader import TaskLoader
from outbound_eval.dataset.store import DataStore
from outbound_eval.scenarios.generator import ScenarioGenerator
from outbound_eval.agent.outbound_agent import OutboundAgent
from outbound_eval.agent.llm.openai_client import OpenAIClient
from outbound_eval.simulator.llm_simulator import UserSimulator
from outbound_eval.judge.engine import JudgeEngine
from outbound_eval.benchmark.pipeline import EvalPipeline
from outbound_eval.dashboard.api import run_server


@click.group()
def cli():
    """Outbound Agent Evaluation Platform CLI."""
    pass


@cli.command()
@click.option("--task", required=True, help="Task ID to run")
@click.option("--scenarios", default=5, help="Number of scenarios")
def run_eval(task: str, scenarios: int):
    """Run evaluation for a single task."""
    click.echo(f"Running evaluation for task: {task}")

    # Load task
    loader = TaskLoader()
    task_obj = loader.load(task)

    # Generate scenarios
    generator = ScenarioGenerator()
    scenario_list = generator.generate(task_obj, num_scenarios=scenarios)

    click.echo(f"Generated {len(scenario_list)} scenarios")

    # Run pipeline
    pipeline = EvalPipeline()
    results = pipeline.run(task_obj, scenario_list)

    click.echo(f"Completed {len(results)} evaluations")

    # Print summary
    passed = sum(1 for r in results if r.get("passed", False))
    success_rate = (passed / len(results) * 100) if results else 0

    click.echo(f"Success rate: {success_rate:.1f}%")


@cli.command()
@click.option("--task", help="Task ID (optional, runs all if not specified)")
@click.option("--scenarios", default=10, help="Scenarios per task")
def batch_eval(task: str | None, scenarios: int):
    """Run batch evaluation."""
    loader = TaskLoader()
    store = DataStore()

    if task:
        tasks = [loader.load(task)]
    else:
        tasks = loader.load_all()

    click.echo(f"Running batch evaluation for {len(tasks)} tasks")

    generator = ScenarioGenerator()
    pipeline = EvalPipeline()

    all_results = []

    for task_obj in tasks:
        scenario_list = generator.generate(task_obj, num_scenarios=scenarios)
        results = pipeline.run(task_obj, scenario_list)
        all_results.extend(results)

        # Save results
        for result in results:
            store.save_result(task_obj.task_id, result.get("scenario_id", "unknown"), result)

    passed = sum(1 for r in all_results if r.get("passed", False))
    success_rate = (passed / len(all_results) * 100) if all_results else 0

    click.echo(f"Total evaluations: {len(all_results)}")
    click.echo(f"Overall success rate: {success_rate:.1f}%")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=None, type=int, help="Port to bind (defaults to PORT env or 8000)")
def serve(host: str, port: int | None):
    """Start the dashboard server."""
    import os
    if port is None:
        port = int(os.environ.get("PORT", 8000))
    click.echo(f"Starting dashboard at http://{host}:{port}")
    run_server(host, port)


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--task-id", required=True, help="Task ID for the generated YAML")
@click.option("--output-dir", default=None,
              help="Output directory (defaults to data/benchmarks/tasks/)")
@click.option("--row", default=0, type=int,
              help="For .xlsx files: zero-based content row to read (default 0)")
def parse_instruction_cmd(file: str, task_id: str, output_dir: str | None, row: int):
    """Parse a task-instruction text/.xlsx file and generate an evaluation YAML."""
    from pathlib import Path
    from outbound_eval.dataset.instruction_parser import (
        parse_and_save, parse_excel_file,
    )
    from outbound_eval.dataset.loader import TaskLoader

    src = Path(file)
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "tasks"
    out_path = out_dir / f"{task_id}.yaml"

    if src.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            parsed, task, saved, _raw = parse_excel_file(src, task_id, out_path, row_index=row)
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)
    else:
        text = src.read_text(encoding="utf-8")
        parsed, task, saved = parse_and_save(text, task_id, out_path)

    click.echo(f"Parsed {len(parsed.flow_steps)} flow steps, "
               f"{len(parsed.faq_items)} FAQ items, "
               f"{len(parsed.constraints)} constraints")
    if parsed.errors:
        click.echo("Warnings:", err=True)
        for e in parsed.errors:
            click.echo(f"  - {e}", err=True)

    try:
        _ = TaskLoader().load(task_id) if not output_dir else None
        click.echo(f"Round-trip validation: OK")
    except Exception as e:
        click.echo(f"Round-trip validation FAILED: {e}", err=True)

    click.echo(f"Saved to: {saved}")


@cli.command()
@click.option("--task-id", required=True, help="Task ID to generate report for")
@click.option("--run-id", default=None, help="Optional run ID filter")
@click.option("--output", default=None, help="Output file path (auto-generated if omitted)")
@click.option("--format", "fmt", default="html", type=click.Choice(["html", "md", "pdf"]),
              help="Output format (default html)")
def gen_report(task_id: str, run_id: str | None, output: str | None, fmt: str):
    """Generate a standalone HTML/Markdown/PDF evaluation report from raw result files."""
    from pathlib import Path
    from datetime import datetime
    from outbound_eval.dataset.loader import TaskLoader
    from outbound_eval.dataset.store import DataStore
    from outbound_eval.reporting.report import ReportBuilder
    from outbound_eval.reporting.pdf import PdfRenderError

    loader = TaskLoader()
    store = DataStore()
    try:
        task = loader.load(task_id)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    files = store.list_results(task_id)
    results = [store.load_result(f) for f in files]
    if run_id:
        results = [r for r in results if r.get("run_id") == run_id]
    if not results:
        click.echo("No results found for the given task/run.", err=True)
        raise SystemExit(1)

    effective_run_id = run_id or results[0].get("run_id", "manual")
    rb = ReportBuilder(task, results, run_id=effective_run_id)

    if output:
        out_path = Path(output)
    else:
        reports_dir = (
            Path(__file__).resolve().parent.parent / "data" / "results" / "reports"
        )
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = {"html": "html", "md": "md", "pdf": "pdf"}[fmt]
        out_path = reports_dir / f"{effective_run_id}_{ts}.{ext}"

    try:
        rb.save(out_path, format=fmt)
    except PdfRenderError as e:
        click.echo(f"PDF 导出失败:\n{e}", err=True)
        raise SystemExit(2)

    click.echo(f"Report generated: {out_path}  ({out_path.stat().st_size} bytes)")


@cli.command()
def seed():
    """Initialize dataset with sample tasks."""
    from outbound_eval.dataset.task import EvaluationTask, SuccessCondition, DifficultyLevel

    loader = TaskLoader()

    # Sample task
    task = EvaluationTask(
        task_id="sample_task",
        name="示例任务",
        description="这是一个示例任务",
        skill_name="feimaotui",
        difficulty=DifficultyLevel.MEDIUM,
        success_criteria=[
            SuccessCondition(
                condition_id="test_condition",
                name="测试条件",
                description="测试条件描述",
            )
        ],
    )

    loader.save(task)
    click.echo("Dataset initialized with sample task")


if __name__ == "__main__":
    cli()