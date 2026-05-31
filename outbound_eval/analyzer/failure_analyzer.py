"""Failure analyzer for error attribution."""

from enum import Enum
from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """Types of failures."""

    FLOW_DEVIATION = "flow_deviation"
    CONSTRAINT_VIOLATION = "constraint_violation"
    INTENT_MISUNDERSTANDING = "intent_misunderstanding"
    RECOVERY_FAILURE = "recovery_failure"
    EARLY_TERMINATION = "early_termination"
    FAQ_MISMATCH = "faq_mismatch"


class FailureAnalysis(BaseModel):
    """Analysis of a failure."""

    task_id: str
    failure_type: FailureType
    root_cause: str
    affected_metrics: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class FailureAnalyzer:
    """Analyzes failures and generates improvement suggestions."""

    # Mapping from failure reasons to failure types
    REASON_TYPE_MAP = {
        "流程执行不完整": FailureType.FLOW_DEVIATION,
        "未满足足够的成功条件": FailureType.FLOW_DEVIATION,
        "状态记录不一致": FailureType.INTENT_MISUNDERSTANDING,
        "话术合规性不达标": FailureType.CONSTRAINT_VIOLATION,
        "异常恢复能力不足": FailureType.RECOVERY_FAILURE,
        "对话过早终止": FailureType.EARLY_TERMINATION,
        "答非所问": FailureType.FAQ_MISMATCH,
    }

    # Root cause templates
    ROOT_CAUSE_TEMPLATES = {
        FailureType.FLOW_DEVIATION: "Agent 未按预期流程执行，可能跳过了关键步骤",
        FailureType.CONSTRAINT_VIOLATION: "Agent 违反了话术约束规则（长度/语气/禁用词）",
        FailureType.INTENT_MISUNDERSTANDING: "Agent 误解了用户意图，状态记录与实际不符",
        FailureType.RECOVERY_FAILURE: "Agent 在用户拒绝/打断后未能有效恢复对话",
        FailureType.EARLY_TERMINATION: "Agent 在任务完成前过早挂断",
        FailureType.FAQ_MISMATCH: "Agent 的回答与用户问题不匹配",
    }

    # Suggestion templates
    SUGGESTION_TEMPLATES = {
        FailureType.FLOW_DEVIATION: [
            "检查并优化 Prompt 中的流程说明",
            "增加步骤检查机制，确保关键步骤被执行",
        ],
        FailureType.CONSTRAINT_VIOLATION: [
            "加强约束规则的 Prompt 说明",
            "增加回复长度检测和截断逻辑",
            "添加禁用词过滤",
        ],
        FailureType.INTENT_MISUNDERSTANDING: [
            "优化意图识别 Prompt",
            "增加用户确认步骤",
            "记录并验证关键状态变化",
        ],
        FailureType.RECOVERY_FAILURE: [
            "增加被打断后的恢复话术",
            "增加用户拒绝后的挽留策略",
            "优化情绪识别和安抚能力",
        ],
        FailureType.EARLY_TERMINATION: [
            "增加任务完成度检查",
            "确保所有关键信息传达后再结束",
        ],
        FailureType.FAQ_MISMATCH: [
            "优化 FAQ 匹配逻辑",
            "增加问题澄清步骤",
        ],
    }

    def analyze(
        self,
        judge_result: dict,
        dialogue_history: list[dict],
    ) -> list[FailureAnalysis]:
        """Analyze failures from judge results.

        Args:
            judge_result: Judge result dict
            dialogue_history: Conversation history

        Returns:
            List of failure analyses
        """
        if judge_result.get("passed", False):
            return []

        analyses: list[FailureAnalysis] = []
        failure_reasons = judge_result.get("failure_reasons", [])

        # Categorize failures
        categorized: dict[FailureType, list[str]] = {}

        for reason in failure_reasons:
            failure_type = self._categorize_failure(reason)
            if failure_type not in categorized:
                categorized[failure_type] = []
            categorized[failure_type].append(reason)

        # Generate analysis for each category
        for failure_type, reasons in categorized.items():
            analysis = FailureAnalysis(
                task_id=judge_result.get("task_id", ""),
                failure_type=failure_type,
                root_cause=self.ROOT_CAUSE_TEMPLATES.get(failure_type, "未知原因"),
                affected_metrics=self._get_affected_metrics(failure_type),
                suggestions=self.SUGGESTION_TEMPLATES.get(failure_type, []),
            )
            analyses.append(analysis)

        return analyses

    def _categorize_failure(self, reason: str) -> FailureType:
        """Categorize a failure reason.

        Args:
            reason: Failure reason text

        Returns:
            FailureType
        """
        for key, failure_type in self.REASON_TYPE_MAP.items():
            if key in reason:
                return failure_type

        # Default fallback
        return FailureType.FLOW_DEVIATION

    def _get_affected_metrics(self, failure_type: FailureType) -> list[str]:
        """Get metrics affected by failure type.

        Args:
            failure_type: Type of failure

        Returns:
            List of affected metric names
        """
        metric_map = {
            FailureType.FLOW_DEVIATION: ["flow_adherence"],
            FailureType.CONSTRAINT_VIOLATION: ["compliance"],
            FailureType.INTENT_MISUNDERSTANDING: ["state_tracking", "task_success"],
            FailureType.RECOVERY_FAILURE: ["recovery"],
            FailureType.EARLY_TERMINATION: ["task_success", "flow_adherence"],
            FailureType.FAQ_MISMATCH: ["task_success"],
        }

        return metric_map.get(failure_type, [])

    def aggregate_failures(
        self,
        all_analyses: list[FailureAnalysis],
    ) -> dict:
        """Aggregate failures across all evaluations.

        Args:
            all_analyses: List of all failure analyses

        Returns:
            Aggregated failure statistics
        """
        stats: dict[FailureType, int] = {}

        for analysis in all_analyses:
            stats[analysis.failure_type] = stats.get(analysis.failure_type, 0) + 1

        # Sort by count
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_failures": len(all_analyses),
            "failure_distribution": dict(sorted_stats),
            "most_common_failure": sorted_stats[0][0] if sorted_stats else None,
        }