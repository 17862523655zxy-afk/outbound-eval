"""LLM-based evaluation with confidence."""

import json
import re
from typing import Optional, Literal
from pydantic import BaseModel, Field
from outbound_eval.agent.llm.base import LLMClient
from outbound_eval.judge.confidence import ConfidenceEstimator


class LLMJudgeResult(BaseModel):
    """Result of LLM-based evaluation."""

    metric_name: str
    score: float
    confidence: float = Field(description="0.0-1.0 confidence level")
    confidence_level: Literal["high", "medium", "low"] = "medium"
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    is_reliable: bool = True
    needs_review: bool = False


class CriterionEval(BaseModel):
    """Result of evaluating a single success/failure condition with LLM.

    Returned by :meth:`LLMJudge.evaluate_condition` and surfaced to the
    dashboard so each criterion is explainable end-to-end.
    """

    satisfied: bool = False
    score: float = 0.0
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    method: str = "llm"  # one of: "llm" | "rule" | "hybrid" | "rule_fallback"
    error: Optional[str] = None


class LLMJudge:
    """LLM-based judge for complex evaluations."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        judge_model: Optional[str] = None,
    ):
        """Initialize the judge.

        Args:
            llm_client: LLM client
            judge_model: Model name for judge
        """
        self.llm_client = llm_client
        self.judge_model = judge_model or "claude-sonnet-4-20250514"
        self.confidence_estimator = ConfidenceEstimator()

    def evaluate_naturalness(
        self, dialogue_history: list[dict[str, object]]
    ) -> LLMJudgeResult:
        """Evaluate conversation naturalness.

        Args:
            dialogue_history: Conversation history

        Returns:
            Evaluation result with confidence
        """
        # Build evaluation prompt
        prompt = self._build_naturalness_prompt(dialogue_history)

        # Call LLM
        response = self._call_judge_llm(prompt)

        # Parse result
        score = self._parse_score(response)
        confidence = self.confidence_estimator.estimate_from_response(response)

        return LLMJudgeResult(
            metric_name="naturalness",
            score=score,
            confidence=confidence.confidence,
            confidence_level=confidence.confidence_level,
            reasoning=response.get("reasoning", ""),
            evidence=response.get("evidence", []),
            is_reliable=confidence.confidence_level != "low",
            needs_review=confidence.confidence_level == "low",
        )

    def evaluate_recovery(
        self, interruption_context: dict, recovery_response: str
    ) -> LLMJudgeResult:
        """Evaluate recovery ability.

        Args:
            interruption_context: Context of the interruption
            recovery_response: Agent's recovery response

        Returns:
            Evaluation result with confidence
        """
        prompt = self._build_recovery_prompt(interruption_context, recovery_response)
        response = self._call_judge_llm(prompt)

        score = self._parse_score(response)
        confidence = self.confidence_estimator.estimate_from_response(response)

        return LLMJudgeResult(
            metric_name="recovery",
            score=score,
            confidence=confidence.confidence,
            confidence_level=confidence.confidence_level,
            reasoning=response.get("reasoning", ""),
            evidence=response.get("evidence", []),
            is_reliable=confidence.confidence_level != "low",
            needs_review=confidence.confidence_level == "low",
        )

    def _build_naturalness_prompt(self, dialogue_history: list[dict[str, object]]) -> str:
        """Build prompt for naturalness evaluation."""
        turns_text = "\n".join(
            f"{t['role']}: {t['content']}" for t in dialogue_history
        )

        return f"""评估以下对话的自然度：

{turns_text}

请评估：
1. 回复是否口语化、自然
2. 是否像真实电话对话
3. 语气是否恰当

请用 JSON 格式返回：
{{
    "score": 0-100 的分数,
    "reasoning": "评估理由",
    "evidence": ["具体证据1", "具体证据2"]
}}"""

    def evaluate_condition(
        self,
        dialogue_history: list[dict[str, object]],
        condition_name: str,
        condition_description: str,
        priority: str = "P1",
    ) -> CriterionEval:
        """Evaluate whether a single success/failure condition is satisfied.

        This is the entry point used by ``JudgeEngine`` for ``check_type:
        "llm"`` and as the LLM half of ``check_type: "hybrid"``.

        Returns a :class:`CriterionEval` with satisfied/score/reasoning/evidence
        fields. Never raises: on any error the result has ``error`` set,
        ``method = "rule_fallback"``, ``satisfied = False``, ``score = 0.0``.
        """
        if not self.llm_client:
            return CriterionEval(
                satisfied=False,
                score=0.0,
                reasoning="LLM 客户端未配置，无法进行 LLM 评估",
                evidence=[],
                method="rule_fallback",
                error="no_llm_client",
            )

        prompt = self._build_condition_judge_prompt(
            dialogue_history, condition_name, condition_description, priority,
        )
        try:
            raw = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            parsed = self._parse_condition_response(raw)
            return CriterionEval(
                satisfied=bool(parsed.get("satisfied", False)),
                score=max(0.0, min(1.0, float(parsed.get("score", 0.0)))),
                reasoning=str(parsed.get("reasoning", "")),
                evidence=list(parsed.get("evidence", []))[:5],
                method="llm",
            )
        except Exception as e:
            return CriterionEval(
                satisfied=False,
                score=0.0,
                reasoning=f"LLM judge 调用失败: {e}",
                evidence=[],
                method="rule_fallback",
                error=str(e),
            )

    def _build_condition_judge_prompt(
        self,
        dialogue_history: list[dict[str, object]],
        condition_name: str,
        condition_description: str,
        priority: str,
    ) -> str:
        """Build a strict JSON-output prompt for single-criterion judging."""
        turns_text = "\n".join(
            f"[t{i+1}] {t.get('role', '?')}: {t.get('content', '')}"
            for i, t in enumerate(dialogue_history)
        )
        return f"""你是一名严格的外呼质量评估员。请判断 Agent 在以下对话中是否满足给出的条件。

【条件名称】{condition_name}
【条件说明】{condition_description}
【优先级】{priority}（P0 = 必须满足）

【对话历史】（按轮次编号，t1 为开场）
{turns_text}

请基于 Agent 的实际发言判断，不要给同情分。如果条件是关于"用户问某问题时 Agent 是否正确回答"，请检查对话中是否出现该问题及 Agent 的对应回复。

**只输出 JSON**，格式严格如下，不要任何额外文字：
{{
    "satisfied": true 或 false,
    "score": 0.0~1.0 的满足度（satisfied=true 时通常 ≥ 0.7，satisfied=false 时通常 ≤ 0.3）,
    "reasoning": "用一句话说明判断依据，30 字以内",
    "evidence": ["[t3] agent 说 '...'", "[t5] agent 提到 '...'"]  // 引用具体轮次，0~5 条
}}"""

    def _parse_condition_response(self, raw: str) -> dict:
        """Parse a possibly-noisy LLM JSON response.

        Tries: code-fenced JSON → raw JSON object → first {...} span → empty.
        """
        if raw is None:
            return {"satisfied": False, "score": 0.0, "reasoning": "", "evidence": []}

        text = raw.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        try:
            return json.loads(text.strip())
        except Exception:
            pass

        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return {
            "satisfied": False,
            "score": 0.0,
            "reasoning": "无法解析 LLM 响应",
            "evidence": [],
        }

    def _build_recovery_prompt(
        self, context: dict, response: str
    ) -> str:
        """Build prompt for recovery evaluation."""
        return f"""评估 Agent 在被打断后的恢复能力：

中断情况：{context.get('interruption', '未知')}
Agent 回复：{response}

请评估：
1. Agent 是否承认被打断
2. 是否自然地恢复对话
3. 是否有效处理了中断

请用 JSON 格式返回：
{{
    "score": 0-100 的分数,
    "reasoning": "评估理由",
    "evidence": ["具体证据1", "具体证据2"]
}}"""

    def _call_judge_llm(self, prompt: str) -> dict:
        """Call LLM for evaluation."""
        if not self.llm_client:
            # Return default result if no LLM client
            return {
                "score": 75.0,
                "reasoning": "No LLM client configured, using default score",
                "evidence": [],
            }

        try:
            import json

            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            # Try to parse JSON response
            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except Exception as e:
            return {
                "score": 75.0,
                "reasoning": f"Error parsing LLM response: {str(e)}",
                "evidence": [],
            }

    def _parse_score(self, response: dict) -> float:
        """Parse score from LLM response."""
        score = response.get("score", 75.0)
        return float(max(0.0, min(100.0, score)))