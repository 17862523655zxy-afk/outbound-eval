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
@click.option("--port", default=8000, help="Port to bind")
def serve(host: str, port: int):
    """Start the dashboard server."""
    click.echo(f"Starting dashboard at http://{host}:{port}")
    run_server(host, port)


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