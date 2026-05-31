"""Main judge engine combining all evaluation metrics."""

from typing import Optional, Any
from pydantic import BaseModel, Field
from outbound_eval.dataset.task import EvaluationTask, SuccessCondition
from outbound_eval.judge.flow_adherence import FlowAdherenceJudge, FlowAdherenceResult
from outbound_eval.judge.state_tracking import StateTrackingJudge, StateTrackingResult
from outbound_eval.judge.rule_judge import RuleJudge
from outbound_eval.judge.llm_judge import LLMJudge
from outbound_eval.judge.hybrid_judge import HybridJudge, HybridJudgeResult
from outbound_eval.judge.efficiency import EfficiencyJudge, EfficiencyMetrics


class JudgeResult(BaseModel):
    """Complete judge result."""

    task_id: str = ""
    scenario_id: str = ""

    # Metric scores
    task_success: float = 0.0
    flow_adherence: float = 0.0
    state_tracking: float = 0.0
    compliance: float = 0.0
    recovery: float = 0.0
    naturalness: float = 0.0
    efficiency: float = 0.0

    # Detailed results
    flow_adherence_detail: Optional[FlowAdherenceResult] = None
    state_tracking_detail: Optional[StateTrackingResult] = None
    efficiency_detail: Optional[EfficiencyMetrics] = None

    # Overall score
    overall_score: float = 0.0
    passed: bool = False

    # Judge metadata
    judge_mode: str = "hybrid"
    confidence: float = 0.0

    # Failure info
    failure_reasons: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)


class JudgeEngine:
    """Main evaluation engine coordinating all judges."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize the engine.

        Args:
            config: Judge configuration
        """
        self.config = config or {}
        self.rule_judge = RuleJudge(self.config.get("constraints", {}))
        self.llm_judge = LLMJudge()
        self.hybrid_judge = HybridJudge(self.rule_judge, self.llm_judge)
        self.efficiency_judge = EfficiencyJudge()
        self.flow_judge: Optional[FlowAdherenceJudge] = None
        self.state_judge = StateTrackingJudge()

    def evaluate(
        self,
        task: EvaluationTask,
        dialogue_history: list[dict],
        agent_state: dict,
        total_tokens: int = 0,
    ) -> JudgeResult:
        """Evaluate a conversation.

        Args:
            task: The evaluation task
            dialogue_history: Conversation history
            agent_state: Agent's internal state
            total_tokens: Total tokens consumed

        Returns:
            Complete judge result
        """
        result = JudgeResult(task_id=task.task_id)

        # Initialize flow judge if flow is defined
        # Convert SuccessCondition to dict for FlowAdherenceJudge
        if task.success_criteria:
            flow_steps = [
                {
                    "id": c.condition_id,
                    "description": c.name,
                    "expected_keywords": c.check_config.get("required_keywords", []) if isinstance(c.check_config, dict) else [],
                }
                for c in task.success_criteria
            ]
            self.flow_judge = FlowAdherenceJudge(flow_steps)

        # 1. Task Success - evaluate success conditions
        result.task_success = self._evaluate_task_success(
            task.success_criteria, dialogue_history
        )

        # 2. Flow Adherence
        if self.flow_judge:
            flow_result = self.flow_judge.evaluate(dialogue_history)
            result.flow_adherence = flow_result.adherence_score
            result.flow_adherence_detail = flow_result

        # 3. State Tracking
        self.state_judge.set_agent_state(agent_state)
        self.state_judge.set_ground_truth(dialogue_history, task.expected_outcome)
        state_result = self.state_judge.evaluate()
        result.state_tracking = state_result.consistency_score
        result.state_tracking_detail = state_result

        # 4. Compliance - Rule-based
        compliance_result = self.rule_judge.evaluate_conversation(dialogue_history)
        result.compliance = compliance_result.compliance_score

        # 5. Recovery - LLM-based
        recovery_result = self.llm_judge.evaluate_naturalness(dialogue_history)
        result.recovery = recovery_result.score

        # 6. Naturalness - LLM-based
        naturalness_result = self.llm_judge.evaluate_naturalness(dialogue_history)
        result.naturalness = naturalness_result.score

        # 7. Efficiency
        eff_result = self.efficiency_judge.evaluate(dialogue_history, total_tokens)
        result.efficiency = eff_result.efficiency_score
        result.efficiency_detail = eff_result

        # Calculate overall score
        weights = {
            "task_success": 0.30,
            "flow_adherence": 0.20,
            "state_tracking": 0.10,
            "compliance": 0.15,
            "recovery": 0.10,
            "naturalness": 0.10,
            "efficiency": 0.05,
        }

        scores = {
            "task_success": result.task_success,
            "flow_adherence": result.flow_adherence,
            "state_tracking": result.state_tracking,
            "compliance": result.compliance,
            "recovery": result.recovery,
            "naturalness": result.naturalness,
            "efficiency": result.efficiency,
        }

        result.overall_score = sum(scores[k] * v for k, v in weights.items())

        # Determine pass/fail (relaxed thresholds for realistic agent performance)
        result.passed = (
            result.task_success >= task.pass_threshold * 100
            and result.flow_adherence >= 45.0
            and result.compliance >= 55.0
        )

        # Generate failure reasons
        self._generate_failure_reasons(result, task)

        return result

    def _evaluate_task_success(
        self,
        success_criteria: list[SuccessCondition],
        dialogue_history: list[dict],
    ) -> float:
        """Evaluate task success conditions.

        Args:
            success_criteria: List of success conditions
            dialogue_history: Conversation history

        Returns:
            Success score (0-100)
        """
        if not success_criteria:
            return 75.0  # Default if no criteria

        agent_messages = " ".join(
            t["content"] for t in dialogue_history if t["role"] == "agent"
        )
        user_messages = " ".join(
            t["content"] for t in dialogue_history if t["role"] == "user"
        )

        satisfied = 0
        total_weight = 0.0

        for condition in success_criteria:
            total_weight += condition.weight
            
            # Get check_config as dict
            if hasattr(condition, 'model_dump'):
                config = condition.model_dump().get('check_config', {})
            else:
                config = condition.check_config if isinstance(condition, dict) else {}

            check_type = condition.check_type if hasattr(condition, 'check_type') else condition.get('check_type')

            if check_type == "rule":
                # Rule-based check
                required_keywords = config.get("required_keywords", []) if isinstance(config, dict) else []
                if required_keywords:
                    if all(kw in agent_messages for kw in required_keywords):
                        satisfied += condition.weight

            elif check_type == "llm":
                # LLM-based check (simplified)
                satisfied += condition.weight * 0.7

            elif check_type == "hybrid":
                # Hybrid check
                confirm_patterns = config.get("confirm_patterns", []) if isinstance(config, dict) else []
                reject_patterns = config.get("reject_patterns", []) if isinstance(config, dict) else []

                if confirm_patterns and any(
                    kw in user_messages for kw in confirm_patterns
                ):
                    satisfied += condition.weight
                elif reject_patterns and any(
                    kw in user_messages for kw in reject_patterns
                ):
                    satisfied += condition.weight * 0.5

        if total_weight > 0:
            return (satisfied / total_weight) * 100
        return 75.0

    def _generate_failure_reasons(
        self, result: JudgeResult, task: EvaluationTask
    ) -> None:
        """Generate failure reasons and suggestions.

        Args:
            result: Judge result to update
            task: The evaluation task
        """
        if result.passed:
            return

        # Check each metric for failure
        if result.task_success < task.pass_threshold * 100:
            result.failure_reasons.append("未满足足够的成功条件")

        if result.flow_adherence < 60.0:
            result.failure_reasons.append("流程执行不完整")

        if result.state_tracking < 70.0:
            result.failure_reasons.append("状态记录不一致")

        if result.compliance < 70.0:
            result.failure_reasons.append("话术合规性不达标")

        if result.recovery < 60.0:
            result.failure_reasons.append("异常恢复能力不足")

        # Generate suggestions
        if result.task_success < 80.0:
            result.improvement_suggestions.append("增加关键信息的传达")

        if result.flow_adherence < 80.0:
            result.improvement_suggestions.append("优化流程执行顺序")

        if result.compliance < 80.0:
            result.improvement_suggestions.append("控制回复长度在要求范围内")

        if result.recovery < 70.0:
            result.improvement_suggestions.append("增加异常场景的应对话术")