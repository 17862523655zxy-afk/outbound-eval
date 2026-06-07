"""Failure analyzer for error attribution."""

from enum import Enum
from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """Types of failures (PRD-aligned classification)."""

    INTENT_RECOGNITION_ERROR = "intent_recognition_error"
    FLOW_VIOLATION = "flow_violation"
    KNOWLEDGE_ERROR = "knowledge_error"
    COMPLIANCE_VIOLATION = "compliance_violation"
    MISSING_INFORMATION = "missing_information"
    WEAK_RETENTION = "weak_retention"
    PREMATURE_ENDING = "premature_ending"
    OTHER = "other"


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
        "流程执行不完整": FailureType.FLOW_VIOLATION,
        "未满足足够的成功条件": FailureType.FLOW_VIOLATION,
        "存在未满足的关键成功条件": FailureType.FLOW_VIOLATION,
        "状态记录不一致": FailureType.INTENT_RECOGNITION_ERROR,
        "话术合规性不达标": FailureType.COMPLIANCE_VIOLATION,
        "异常恢复能力不足": FailureType.WEAK_RETENTION,
        "对话过早终止": FailureType.PREMATURE_ENDING,
        "答非所问": FailureType.KNOWLEDGE_ERROR,
        "增加关键信息的传达": FailureType.MISSING_INFORMATION,
        "触发失败条件": FailureType.COMPLIANCE_VIOLATION,
    }

    # Root cause templates
    ROOT_CAUSE_TEMPLATES = {
        FailureType.INTENT_RECOGNITION_ERROR: "Agent 误解了用户意图，状态记录与实际不符",
        FailureType.FLOW_VIOLATION: "Agent 未按预期流程执行，可能跳过了关键步骤",
        FailureType.KNOWLEDGE_ERROR: "Agent 提供了错误的业务知识或信息",
        FailureType.COMPLIANCE_VIOLATION: "Agent 违反了话术约束规则（长度/语气/禁用词）",
        FailureType.MISSING_INFORMATION: "Agent 遗漏了必须传达的关键信息",
        FailureType.WEAK_RETENTION: "Agent 在用户拒绝/打断后未能有效恢复对话或挽留",
        FailureType.PREMATURE_ENDING: "Agent 在任务完成前过早挂断",
        FailureType.OTHER: "存在未分类的失败原因",
    }

    SUGGESTION_TEMPLATES = {
        FailureType.INTENT_RECOGNITION_ERROR: [
            "优化意图识别 Prompt",
            "增加用户确认步骤",
            "记录并验证关键状态变化",
        ],
        FailureType.FLOW_VIOLATION: [
            "检查并优化 Prompt 中的流程说明",
            "增加步骤检查机制，确保关键步骤被执行",
        ],
        FailureType.KNOWLEDGE_ERROR: [
            "更新知识库内容",
            "增加知识校验机制",
            "优化 FAQ 匹配逻辑",
        ],
        FailureType.COMPLIANCE_VIOLATION: [
            "加强约束规则的 Prompt 说明",
            "增加回复长度检测和截断逻辑",
            "添加禁用词过滤",
        ],
        FailureType.MISSING_INFORMATION: [
            "增加关键信息检查清单",
            "优化信息传达顺序",
        ],
        FailureType.WEAK_RETENTION: [
            "增加被打断后的恢复话术",
            "增加用户拒绝后的挽留策略",
            "优化情绪识别和安抚能力",
        ],
        FailureType.PREMATURE_ENDING: [
            "增加任务完成度检查",
            "确保所有关键信息传达后再结束",
        ],
        FailureType.OTHER: [
            "分析对话日志定位具体问题",
            "增加日志覆盖率",
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
        return FailureType.FLOW_VIOLATION

    def _get_affected_metrics(self, failure_type: FailureType) -> list[str]:
        """Get metrics affected by failure type.

        Args:
            failure_type: Type of failure

        Returns:
            List of affected metric names
        """
        metric_map = {
            FailureType.INTENT_RECOGNITION_ERROR: ["state_tracking", "task_success"],
            FailureType.FLOW_VIOLATION: ["flow_adherence", "task_success"],
            FailureType.KNOWLEDGE_ERROR: ["task_success", "compliance"],
            FailureType.COMPLIANCE_VIOLATION: ["compliance"],
            FailureType.MISSING_INFORMATION: ["task_success", "flow_adherence"],
            FailureType.WEAK_RETENTION: ["recovery", "task_success"],
            FailureType.PREMATURE_ENDING: ["task_success", "flow_adherence"],
            FailureType.OTHER: ["task_success"],
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