"""LLM-based evaluation with confidence."""

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
        self, dialogue_history: list[dict]
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

    def _build_naturalness_prompt(self, dialogue_history: list[dict]) -> str:
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