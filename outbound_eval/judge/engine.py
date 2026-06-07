"""Main judge engine combining all evaluation metrics."""

from typing import TYPE_CHECKING, cast
from pydantic import BaseModel, Field
from outbound_eval.dataset.task import EvaluationTask, SuccessCondition, FailureCondition
from outbound_eval.judge.flow_adherence import FlowAdherenceJudge, FlowAdherenceResult
from outbound_eval.judge.state_tracking import StateTrackingJudge, StateTrackingResult
from outbound_eval.judge.rule_judge import RuleJudge
from outbound_eval.judge.llm_judge import LLMJudge
from outbound_eval.judge.hybrid_judge import HybridJudge
from outbound_eval.judge.efficiency import EfficiencyJudge, EfficiencyMetrics
from outbound_eval.gold.comparator import GoldComparator, GoldComparisonResult

if TYPE_CHECKING:
    from outbound_eval.agent.llm.base import LLMClient


# A single turn in a dialogue history.
DialogueTurn = dict[str, object]


class CriterionDetail(BaseModel):
    """Per-condition explainability record.

    Populated for every success/failure condition evaluated, regardless of
    whether the check is rule, LLM, or hybrid. Surfaced to the dashboard
    so each score is traceable back to a concrete method + reasoning +
    evidence quote.
    """

    condition_id: str
    name: str
    priority: str
    check_type: str
    satisfied: bool
    score: float = Field(description="0.0-1.0 normalized")
    method: str  # "rule" | "llm" | "hybrid" | "rule_fallback"
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    error: str | None = None


class JudgeWeights(BaseModel):
    """Configurable weights for judge dimensions."""

    task_success: float = 0.40
    flow_adherence: float = 0.20
    state_tracking: float = 0.10
    compliance: float = 0.10
    recovery: float = 0.00
    naturalness: float = 0.10
    efficiency: float = 0.10

    # v2: thresholds for triggering improvement suggestions (configurable)
    weak_dimension_threshold: float = Field(
        default=80.0,
        description="当某维度低于此值时触发改进建议（0-100）",
    )

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "JudgeWeights":
        """Create from dict with fallbacks to defaults."""
        return cls(
            task_success=d.get("task_success", 0.40),
            flow_adherence=d.get("flow_adherence", 0.20),
            state_tracking=d.get("state_tracking", 0.10),
            compliance=d.get("compliance", 0.10),
            recovery=d.get("recovery", 0.00),
            naturalness=d.get("naturalness", 0.10),
            efficiency=d.get("efficiency", 0.10),
            weak_dimension_threshold=d.get("weak_dimension_threshold", 80.0),
        )


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
    flow_adherence_detail: FlowAdherenceResult | None = None
    state_tracking_detail: StateTrackingResult | None = None
    efficiency_detail: EfficiencyMetrics | None = None

    # State tracking path
    state_path: list[str] = Field(default_factory=list)

    # Overall score
    overall_score: float = 0.0
    passed: bool = False

    # P0 tracking
    p0_satisfied: bool = True
    p0_total: int = 0
    p0_passed: int = 0

    # Failure tracking
    failure_violations: list[str] = Field(default_factory=list)

    # Judge metadata
    judge_mode: str = "hybrid"
    confidence: float = 0.0

    # Failure info
    failure_reasons: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)

    # Per-condition explainability
    criterion_details: list[CriterionDetail] = Field(default_factory=list)

    # Gold conversation comparison
    gold_comparison: GoldComparisonResult | None = None


class JudgeEngine:
    """Main evaluation engine coordinating all judges."""

    def __init__(
        self,
        config: dict[str, object] | None = None,
        llm_client: "LLMClient | None" = None,
    ):
        """Initialize the engine.

        Args:
            config: Judge configuration
            llm_client: Optional LLM client; when provided, ``LLMJudge`` is
                wired with it so ``check_type: llm/hybrid`` actually call
                the model instead of returning a fixed 70% placeholder.
        """
        self.config: dict[str, object] = config or {}
        self.rule_judge: RuleJudge = RuleJudge(cast(dict[str, object], self.config.get("constraints", {})))
        self.llm_judge: LLMJudge = LLMJudge(llm_client=llm_client)
        self.hybrid_judge: HybridJudge = HybridJudge(self.rule_judge, self.llm_judge)
        self.efficiency_judge: EfficiencyJudge = EfficiencyJudge()
        self.flow_judge: FlowAdherenceJudge | None = None
        self.state_judge: StateTrackingJudge = StateTrackingJudge()
        self.gold_comparator: GoldComparator = GoldComparator()

    def evaluate(
        self,
        task: EvaluationTask,
        dialogue_history: list[DialogueTurn],
        agent_state: dict[str, object],
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
            flow_steps: list[dict[str, object]] = [
                {
                    "id": c.condition_id,
                    "description": c.name,
                    "expected_keywords": [
                        str(k) for k in cast("list[object]", c.check_config.get("required_keywords", []))
                    ]
                    if isinstance(c.check_config.get("required_keywords"), list)
                    else [],
                }
                for c in task.success_criteria
            ]
            self.flow_judge = FlowAdherenceJudge(flow_steps)

        # 1. Task Success - evaluate success conditions
        result.task_success = self._evaluate_task_success(
            task.success_criteria, dialogue_history, result
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
        result.state_path = self.state_judge.extract_state_path(dialogue_history)

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

        # 8. Check failure criteria
        result.failure_violations = self._evaluate_failure_criteria(
            task.failure_criteria, dialogue_history
        )

        # Calculate overall score using configurable weights
        if isinstance(task.judge_weights, JudgeWeights):
            weights = task.judge_weights
        elif task.judge_weights:
            weights = JudgeWeights.from_dict(task.judge_weights)
        else:
            weights = JudgeWeights()
        weak_th = weights.weak_dimension_threshold

        scores = {
            "task_success": result.task_success,
            "flow_adherence": result.flow_adherence,
            "state_tracking": result.state_tracking,
            "compliance": result.compliance,
            "recovery": result.recovery,
            "naturalness": result.naturalness,
            "efficiency": result.efficiency,
        }

        result.overall_score = sum(
            scores[k] * getattr(weights, k) for k in scores
        )

        # Determine pass/fail (very lenient — use overall_score as the dominant signal;
        # only fail when score is essentially 0 or critical P0 violations occur)
        # P0 success conditions must ALL be satisfied
        # P0-level failure criteria violations → hard fail
        # P1/P2 failure criteria violations → recorded in failure_reasons only
        p0_failure_violations = [
            v for v in result.failure_violations if v.startswith("[P0]")
        ]
        result.passed = (
            result.p0_satisfied
            and len(p0_failure_violations) == 0
            and result.overall_score >= task.pass_threshold * 100
        )

        # Generate failure reasons
        self._generate_failure_reasons(result, task, weak_th)

        # Gold conversation comparison (best-effort, never raises)
        try:
            result.gold_comparison = self.gold_comparator.find_and_compare(
                dialogue_history,
                task.task_id,
            )
        except Exception:
            result.gold_comparison = None

        return result

    def _evaluate_task_success(
        self,
        success_criteria: list[SuccessCondition],
        dialogue_history: list[DialogueTurn],
        result: JudgeResult | None = None,
    ) -> float:
        """Evaluate task success conditions with P0 priority logic.

        Args:
            success_criteria: List of success conditions
            dialogue_history: Conversation history
            result: Optional JudgeResult to populate p0_satisfied and criterion_details

        Returns:
            Success score (0-100)
        """
        if not success_criteria:
            return 75.0  # Default if no criteria

        agent_messages = " ".join(
            cast(str, t["content"])
            for t in dialogue_history
            if t.get("role") == "agent"
        )
        user_messages = " ".join(
            cast(str, t["content"])
            for t in dialogue_history
            if t.get("role") == "user"
        )

        satisfied = 0.0
        total_weight = 0.0
        p0_satisfied = True
        p0_total = 0
        p0_passed = 0
        details: list[CriterionDetail] = []

        for condition in success_criteria:
            priority = condition.priority
            weight = condition.weight
            total_weight += weight

            # ``check_config`` is a typed dict[str, object]; the values we use
            # are all string lists, so cast through ``object`` at the boundary.
            config: dict[str, object] = condition.check_config
            required_keywords_raw = config.get("required_keywords", [])
            required_keywords: list[str] = (
                [str(x) for x in cast("list[object]", required_keywords_raw)]
                if isinstance(required_keywords_raw, list)
                else []
            )
            check_type = condition.check_type

            method_used = "rule"
            reasoning = ""
            evidence: list[str] = []
            normalized_score = 0.0  # 0.0~1.0
            error_msg: str | None = None

            if check_type == "rule":
                # Rule-based check with continuous scoring (not 0/100 binary)
                if required_keywords:
                    # Per-keyword match → continuous score: N/M (was binary 0/1)
                    hit = [kw for kw in required_keywords if kw in agent_messages]
                    miss = [kw for kw in required_keywords if kw not in agent_messages]
                    total = len(required_keywords)
                    matched = len(hit)
                    normalized_score = (matched / total) if total > 0 else 0.0
                    method_used = "rule"
                    reasoning = (
                        f"规则关键词命中 {matched}/{total}（{hit}）；缺失 {miss}；得分 {normalized_score*100:.1f}"
                    )
                    evidence = [
                        f"[t{i+1}] {(cast(str, t.get('content', '')) or '')[:80].replace(chr(10), ' ')}"
                        for i, t in enumerate(dialogue_history)
                        if t.get("role") == "agent"
                    ][:3]
                    if miss:
                        evidence.append(f"❌ 缺失关键词: {' / '.join(miss)}")
                else:
                    # No rule config → fall back to llm if available
                    eval_result = self.llm_judge.evaluate_condition(
                        dialogue_history,
                        condition.name,
                        condition.description,
                        priority,
                    )
                    method_used = eval_result.method
                    normalized_score = eval_result.score
                    reasoning = eval_result.reasoning or (
                        "无规则配置，LLM 评估" if eval_result.method == "llm"
                        else "无规则配置且 LLM 不可用"
                    )
                    evidence = list(eval_result.evidence)
                    error_msg = eval_result.error

            elif check_type == "llm":
                # Real LLM-based check
                eval_result = self.llm_judge.evaluate_condition(
                    dialogue_history,
                    condition.name,
                    condition.description,
                    priority,
                )
                method_used = eval_result.method
                normalized_score = eval_result.score
                reasoning = eval_result.reasoning
                evidence = list(eval_result.evidence)
                error_msg = eval_result.error

            elif check_type == "hybrid":
                # Rule first: if any confirm_pattern or required_keyword matches, give rule half.
                confirm_patterns_raw = config.get("confirm_patterns", [])
                reject_patterns_raw = config.get("reject_patterns", [])
                confirm_patterns: list[str] = (
                    [str(x) for x in cast("list[object]", confirm_patterns_raw)]
                    if isinstance(confirm_patterns_raw, list)
                    else []
                )
                reject_patterns: list[str] = (
                    [str(x) for x in cast("list[object]", reject_patterns_raw)]
                    if isinstance(reject_patterns_raw, list)
                    else []
                )

                rule_hit_confirm = bool(confirm_patterns) and any(
                    p in user_messages for p in confirm_patterns
                )
                rule_hit_reject = bool(reject_patterns) and any(
                    p in user_messages for p in reject_patterns
                )
                rule_hit_keyword = bool(required_keywords) and all(
                    kw in agent_messages for kw in required_keywords
                )

                if rule_hit_confirm or rule_hit_keyword:
                    rule_score = 1.0
                elif rule_hit_reject:
                    rule_score = 0.5  # partial credit for explicit rejection
                else:
                    rule_score = 0.0

                # Always consult LLM for the other half
                eval_result = self.llm_judge.evaluate_condition(
                    dialogue_history,
                    condition.name,
                    condition.description,
                    priority,
                )
                method_used = f"hybrid(rule={rule_score:.1f},llm={eval_result.method})"
                normalized_score = 0.5 * rule_score + 0.5 * eval_result.score
                reasoning = f"rule={rule_score:.1f} | llm: {eval_result.reasoning}"
                evidence = list(eval_result.evidence)
                error_msg = eval_result.error

            else:
                # Unknown check_type → rule with required_keywords (continuous score)
                if required_keywords:
                    hit = [kw for kw in required_keywords if kw in agent_messages]
                    miss = [kw for kw in required_keywords if kw not in agent_messages]
                    total = len(required_keywords)
                    matched = len(hit)
                    normalized_score = (matched / total) if total > 0 else 0.0
                    reasoning = f"未知 check_type={check_type!r}，按规则处理：命中 {matched}/{total}"
                    if miss:
                        evidence = [f"❌ 缺失关键词: {' / '.join(miss)}"]
                else:
                    normalized_score = 0.0
                    reasoning = f"未知 check_type={check_type!r}，无规则配置"

            condition_satisfied = normalized_score >= 0.7
            satisfied += weight * normalized_score

            details.append(CriterionDetail(
                condition_id=condition.condition_id,
                name=condition.name,
                priority=priority,
                check_type=check_type,
                satisfied=condition_satisfied,
                score=normalized_score,
                method=method_used,
                reasoning=reasoning,
                evidence=evidence,
                error=error_msg,
            ))

            # Track P0
            if priority == "P0":
                p0_total += 1
                if condition_satisfied:
                    p0_passed += 1
                else:
                    p0_satisfied = False

        if result is not None:
            result.p0_satisfied = p0_satisfied
            result.p0_total = p0_total
            result.p0_passed = p0_passed
            result.criterion_details = details

        if total_weight > 0:
            return (satisfied / total_weight) * 100
        return 75.0

    def _evaluate_failure_criteria(
        self,
        failure_criteria: list[FailureCondition],
        dialogue_history: list[DialogueTurn],
    ) -> list[str]:
        """Evaluate failure conditions - things that should NOT happen.

        Args:
            failure_criteria: List of failure conditions
            dialogue_history: Conversation history

        Returns:
            List of violated condition names
        """
        if not failure_criteria:
            return []

        agent_messages = " ".join(
            cast(str, t["content"])
            for t in dialogue_history
            if t.get("role") == "agent"
        )

        violations: list[str] = []
        for condition in failure_criteria:
            config: dict[str, object] = condition.check_config
            check_type = condition.check_type
            priority = condition.priority
            condition_name = condition.name

            violated = False
            if check_type == "rule":
                required_raw = config.get("required_keywords", [])
                prohibited_raw = config.get("prohibited_keywords", [])
                required_keywords: list[str] = (
                    [str(x) for x in cast("list[object]", required_raw)]
                    if isinstance(required_raw, list)
                    else []
                )
                prohibited_keywords: list[str] = (
                    [str(x) for x in cast("list[object]", prohibited_raw)]
                    if isinstance(prohibited_raw, list)
                    else []
                )

                if required_keywords:
                    # If required keywords are NOT found, it's a violation
                    if not all(kw in agent_messages for kw in required_keywords):
                        violated = True
                if prohibited_keywords:
                    # If prohibited keywords ARE found, it's a violation
                    if any(kw in agent_messages for kw in prohibited_keywords):
                        violated = True

            if violated:
                violations.append(f"[{priority}] {condition_name}")

        return violations

    def _generate_failure_reasons(
        self, result: JudgeResult, task: EvaluationTask, weak_th: float = 60.0
    ) -> None:
        """Generate failure reasons and suggestions.

        Args:
            result: Judge result to update
            task: The evaluation task
        """
        if result.passed:
            return

        # Check P0 violations
        if not result.p0_satisfied:
            result.failure_reasons.append("存在未满足的关键成功条件 (P0)")

        # Check failure criteria violations
        for violation in result.failure_violations:
            result.failure_reasons.append(f"触发失败条件: {violation}")

        # Check each metric for failure (aligned with relaxed pass gates)
        if result.task_success < task.pass_threshold * 100:
            result.failure_reasons.append("未满足足够的成功条件")

        if result.flow_adherence < 30.0:
            result.failure_reasons.append("流程执行不完整")

        if result.state_tracking < 50.0:
            result.failure_reasons.append("状态记录不一致")

        if result.compliance < 45.0:
            result.failure_reasons.append("话术合规性不达标")

        if result.recovery < 40.0:
            result.failure_reasons.append("异常恢复能力不足")

        # Generate suggestions
        if not result.p0_satisfied:
            result.improvement_suggestions.append("优先确保关键步骤 (P0) 全部完成")

        if result.failure_violations:
            result.improvement_suggestions.append("避免触发禁用话术和失败条件")

        if result.task_success < weak_th:
            result.improvement_suggestions.append("增加关键信息的传达")

        if result.flow_adherence < weak_th:
            result.improvement_suggestions.append("优化流程执行顺序")

        if result.compliance < weak_th:
            result.improvement_suggestions.append("控制回复长度在要求范围内")

        if result.recovery < 70.0:
            result.improvement_suggestions.append("增加异常场景的应对话术")