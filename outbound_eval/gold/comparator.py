"""Gold-conversation comparator.

Compares an actual ``dialogue_history`` against a :class:`GoldConversation`
and produces a structured :class:`GoldComparisonResult` covering three
orthogonal dimensions:

  1. **coverage** — what fraction of the gold agent's "must-say" content
     also appears in the actual agent messages (Jaccard over Chinese
     2-grams, which is robust to small wording differences).
  2. **sequence_alignment** — how similar the order of agent messages is
     to the gold (SequenceMatcher.ratio over the agent-message list).
  3. **outcome_match** — heuristic check on whether the actual
     conversation ends in a state consistent with the gold's
     ``quality_level``.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from outbound_eval.gold.conversation import GoldConversation
from outbound_eval.gold.loader import GoldConversationLoader


CLOSING_KEYWORDS = (
    "再见", "好的", "注意安全", "没问题", "先这样", "挂了",
    "随时联系", "辛苦了", "拜拜", "晚安",
)
COMMITMENT_KEYWORDS = (
    "可以", "好的", "没问题", "愿意", "上线", "开始", "行",
    "答应", "同意", "是的",
)
REJECTION_KEYWORDS = (
    "不送", "不做了", "退出", "不想", "不跑", "不干", "不签",
    "拒绝", "不同意", "算了",
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GoldComparisonResult(BaseModel):
    """Structured output of :meth:`GoldComparator.compare`."""

    matched_gold_id: Optional[str] = None
    gold_quality: Optional[str] = None

    # Coverage: 0.0~1.0 — fraction of gold agent n-grams present in actual
    coverage: float = 0.0
    # Sequence alignment: 0.0~1.0 — SequenceMatcher ratio of agent messages
    sequence_alignment: float = 0.0
    # Outcome match: True / False / None (unknown)
    outcome_match: Optional[bool] = None

    # Explanatory details
    coverage_detail: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


def _chinese_ngrams(text: str, n: int = 2) -> set[str]:
    """Return a set of character n-grams (default bigrams)."""
    text = (text or "").strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _agent_messages(dialogue_history: list[dict[str, object]]) -> list[str]:
    return [str(t.get("content", "")) for t in dialogue_history if t.get("role") == "agent"]


def _user_messages(dialogue_history: list[dict[str, object]]) -> list[str]:
    return [str(t.get("content", "")) for t in dialogue_history if t.get("role") == "user"]


def _coverage(actual_agent: list[str], gold_agent: list[str]) -> tuple[float, dict]:
    """Jaccard over Chinese bigrams of agent messages."""
    actual_ngrams = set().union(*(_chinese_ngrams(m) for m in actual_agent)) if actual_agent else set()
    gold_ngrams = set().union(*(_chinese_ngrams(m) for m in gold_agent)) if gold_agent else set()
    if not gold_ngrams:
        return 0.0, {"gold_ngrams": 0, "actual_ngrams": 0, "intersection": 0}
    intersection = actual_ngrams & gold_ngrams
    union = actual_ngrams | gold_ngrams
    jaccard = len(intersection) / len(union) if union else 0.0
    recall = len(intersection) / len(gold_ngrams) if gold_ngrams else 0.0
    return jaccard, {
        "gold_ngrams": len(gold_ngrams),
        "actual_ngrams": len(actual_ngrams),
        "intersection": len(intersection),
        "gold_recall": round(recall, 3),
    }


def _sequence_alignment(actual_agent: list[str], gold_agent: list[str]) -> float:
    """SequenceMatcher ratio on agent-message lists."""
    if not actual_agent or not gold_agent:
        return 0.0
    return difflib.SequenceMatcher(a=gold_agent, b=actual_agent).ratio()


def _infer_actual_outcome(dialogue_history: list[dict[str, object]]) -> Optional[str]:
    """Heuristically classify the conversation outcome.

    Returns one of: "commitment", "rejection", "neutral", or None if
    there isn't enough signal.
    """
    user_msgs = _user_messages(dialogue_history)
    if not user_msgs:
        return None
    last_user = user_msgs[-1] or ""
    has_commit = any(kw in m for m in user_msgs for kw in COMMITMENT_KEYWORDS)
    has_reject = any(kw in m for m in user_msgs for kw in REJECTION_KEYWORDS)
    if has_commit and not has_reject:
        return "commitment"
    if has_reject and not has_commit:
        return "rejection"
    if has_commit and has_reject:
        # Tie-break: look at the last user message
        if any(kw in last_user for kw in COMMITMENT_KEYWORDS):
            return "commitment"
        if any(kw in last_user for kw in REJECTION_KEYWORDS):
            return "rejection"
    return "neutral"


def _outcome_match(dialogue_history: list[dict[str, object]], gold: GoldConversation) -> Optional[bool]:
    """Heuristically check whether actual outcome aligns with gold's quality."""
    actual = _infer_actual_outcome(dialogue_history)
    if actual is None:
        return None

    # Heuristic: excellent/gold endings should look like commitment + closing
    agent_msgs = _agent_messages(dialogue_history)
    has_closing = bool(agent_msgs) and any(
        kw in agent_msgs[-1] for kw in CLOSING_KEYWORDS
    )

    if gold.quality_level in ("excellent", "good"):
        return actual == "commitment" and has_closing
    # For lower-quality golds, just check no-commitment doesn't make it worse
    return actual != "rejection"


class GoldComparator:
    """Compare actual conversations against gold references."""

    def __init__(self, library_dir: Optional[str] = None):
        self.loader = GoldConversationLoader(library_dir=library_dir)

    def find_best_gold(
        self,
        dialogue_history: list[dict[str, object]],
        task_id: str,
        persona_type: Optional[str] = None,
    ) -> Optional[GoldConversation]:
        """Find the gold conversation most similar to the actual dialogue."""
        candidates = self.loader.load_for_task(task_id, persona_type=persona_type)
        if not candidates:
            return None
        actual_agent = _agent_messages(dialogue_history)
        if not actual_agent:
            return None

        scored = []
        for gold in candidates:
            gold_agent = [t.content for t in gold.get_agent_turns()]
            score = _sequence_alignment(actual_agent, gold_agent)
            scored.append((score, gold))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def compare(
        self,
        dialogue_history: list[dict[str, object]],
        gold: GoldConversation,
    ) -> GoldComparisonResult:
        actual_agent = _agent_messages(dialogue_history)
        gold_agent = [t.content for t in gold.get_agent_turns()]

        coverage, coverage_detail = _coverage(actual_agent, gold_agent)
        seq_align = _sequence_alignment(actual_agent, gold_agent)
        outcome_ok = _outcome_match(dialogue_history, gold)

        notes: list[str] = []
        if coverage < 0.2:
            notes.append("实际对话与 gold 内容重合度极低（<20%），可能走了完全不同的路径")
        elif coverage < 0.4:
            notes.append("实际对话与 gold 有部分重合，但遗漏较多关键信息")
        if seq_align < 0.3 and coverage >= 0.4:
            notes.append("关键词覆盖尚可，但说话顺序与 gold 差异较大")
        if outcome_ok is False:
            notes.append("对话结果与 gold 的 quality_level 不一致")
        if not notes:
            notes.append("实际对话与 gold 较为一致")

        return GoldComparisonResult(
            matched_gold_id=gold.conversation_id,
            gold_quality=gold.quality_level,
            coverage=round(coverage, 3),
            sequence_alignment=round(seq_align, 3),
            outcome_match=outcome_ok,
            coverage_detail=coverage_detail,
            notes=notes,
        )

    def find_and_compare(
        self,
        dialogue_history: list[dict[str, object]],
        task_id: str,
        persona_type: Optional[str] = None,
    ) -> Optional[GoldComparisonResult]:
        """Convenience: find best gold for the task and compare."""
        gold = self.find_best_gold(dialogue_history, task_id, persona_type)
        if gold is None:
            return None
        return self.compare(dialogue_history, gold)
