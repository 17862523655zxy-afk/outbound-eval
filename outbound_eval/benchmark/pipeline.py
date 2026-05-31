"""Evaluation pipeline."""

import traceback
import time
import uuid
from datetime import datetime
from typing import Optional
from outbound_eval.dataset.task import EvaluationTask
from outbound_eval.dataset.store import DataStore
from outbound_eval.agent.outbound_agent import OutboundAgent
from outbound_eval.agent.llm.openai_client import OpenAIClient
from outbound_eval.simulator.llm_simulator import UserSimulator
from outbound_eval.judge.engine import JudgeEngine
from outbound_eval.scenarios.personas import UserPersona
from outbound_eval.infra.config import settings
from outbound_eval.benchmark.monitor import RunMonitor


class EvalPipeline:
    """Main evaluation pipeline."""

    def __init__(self):
        """Initialize the pipeline."""
        self.llm_client = OpenAIClient()

    def run(
        self,
        task: EvaluationTask,
        scenarios: list[dict],
        run_name: Optional[str] = None,
    ) -> list[dict]:
        """Run evaluation pipeline.

        Args:
            task: Evaluation task
            scenarios: List of scenarios
            run_name: Optional user-defined run name

        Returns:
            List of evaluation results
        """
        store = DataStore()
        monitor = RunMonitor()
        results = []

        # Generate a unique run ID for this batch
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Start monitoring
        monitor.start_run(task.task_id, len(scenarios))

        for scenario in scenarios:
            scenario_id = scenario.get("scenario_id", "unknown")
            try:
                monitor.scenario_started(scenario_id)
                result = self._run_single(task, scenario, run_id, run_name)
                results.append(result)
                # Save result to database
                store.save_result(
                    task.task_id,
                    result.get("scenario_id", ""),
                    result,
                )
                # Update monitor
                monitor.scenario_completed(
                    scenario_id=scenario_id,
                    passed=result.get("passed", False),
                    score=result.get("overall_score", 0.0),
                    turns=len(result.get("dialogue_history", [])),
                    cost=result.get("cost", 0.0),
                    persona_type=result.get("persona_type", ""),
                )
            except Exception as e:
                print(f"Error in scenario {scenario.get('scenario_id', 'unknown')}: {e}")
                traceback.print_exc()
                monitor.scenario_failed(scenario_id, str(e))

        # Finish monitoring
        monitor.finish_run()

        return results

    def _run_single(
        self,
        task: EvaluationTask,
        scenario: dict,
        run_id: str = "",
        run_name: Optional[str] = None,
    ) -> dict:
        """Run evaluation for a single scenario.

        Args:
            task: Evaluation task
            scenario: Scenario dict

        Returns:
            Evaluation result
        """
        start = time.time()

        # Create agent
        agent = OutboundAgent(llm_client=self.llm_client)
        agent.load_skill(task.skill_name, **task.variables)

        # Create user simulator
        persona_data = scenario.get("persona", {})
        persona = UserPersona(**persona_data)
        simulator = UserSimulator(llm_client=self.llm_client)
        simulator.set_persona(persona)

        # Start conversation
        opening = agent.start_conversation()
        user_response = simulator.generate_response(opening, current_turn=1)

        # Multi-turn conversation
        max_turns = settings.max_turns_per_call
        for turn in range(2, max_turns + 1):
            # Agent responds
            agent_response = agent.respond(user_response)

            # Check if should end
            if agent.should_end_conversation() or simulator.should_end_conversation():
                break

            # User responds
            user_response = simulator.generate_response(agent_response, current_turn=turn)

        # Get evaluation data
        dialogue_history = agent.get_conversation_history()
        agent_state = agent.get_current_state()

        # Judge
        judge_engine = JudgeEngine()
        judge_result = judge_engine.evaluate(
            task=task,
            dialogue_history=dialogue_history,
            agent_state=agent_state,
            total_tokens=self.llm_client.total_tokens,
        )

        # Gather enhanced simulator data
        memory_data = None
        if simulator.get_memory():
            memory_data = simulator.get_memory().model_dump()

        state_transitions = []
        if simulator.get_state_machine():
            state_transitions = simulator.get_state_machine().get_transition_log()

        triggered_events = []
        if simulator.event_injector:
            triggered_events = simulator.event_injector.get_triggered_events_log()

        # Build result
        return {
            "run_id": run_id,
            "run_name": run_name or run_id,
            "scenario_id": scenario.get("scenario_id", ""),
            "task_id": task.task_id,
            "persona_type": persona.persona_type.value,
            "difficulty": scenario.get("difficulty", "medium"),
            "dialogue_history": dialogue_history,
            "agent_state": agent_state,
            "overall_score": judge_result.overall_score,
            "passed": judge_result.passed,
            "pass_threshold": task.pass_threshold,
            "task_success": judge_result.task_success,
            "flow_adherence": judge_result.flow_adherence,
            "flow_adherence_detail": judge_result.flow_adherence_detail.model_dump() if judge_result.flow_adherence_detail else None,
            "state_tracking": judge_result.state_tracking,
            "state_tracking_detail": judge_result.state_tracking_detail.model_dump() if judge_result.state_tracking_detail else None,
            "compliance": judge_result.compliance,
            "recovery": judge_result.recovery,
            "naturalness": judge_result.naturalness,
            "efficiency": judge_result.efficiency,
            "efficiency_detail": judge_result.efficiency_detail.model_dump() if judge_result.efficiency_detail else None,
            "failure_reasons": judge_result.failure_reasons,
            "improvement_suggestions": judge_result.improvement_suggestions,
            "total_tokens": self.llm_client.total_tokens,
            "cost": self.llm_client.total_cost,
            # Enhanced fields for Dashboard analysis
            "persona_profile": persona.model_dump(),
            "conversation_memory": memory_data,
            "state_transitions": state_transitions,
            "triggered_events": triggered_events,
            "elapsed_seconds": round(time.time() - start, 1),
        }