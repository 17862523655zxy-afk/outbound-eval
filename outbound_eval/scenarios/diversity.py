"""Response diversity management for user simulator."""

from collections import deque


class ResponseDiversityManager:
    """Manages response expression diversity to avoid repetition."""

    def __init__(self, style_pool: dict[str, list[str]] | None = None):
        """Initialize with optional style pool.

        Args:
            style_pool: Expression pool per intent, e.g.
                {"agree": ["行", "可以试试"], "reject": ["不想跑了"]}
        """
        self.style_pool = style_pool or self._default_pool()
        self.recent_styles: deque[str] = deque(maxlen=3)
        self.recent_keywords: deque[str] = deque(maxlen=5)
        self.recent_full_texts: deque[str] = deque(maxlen=3)

    def _default_pool(self) -> dict[str, list[str]]:
        """Get default expression pool."""
        return {
            "agree": ["行", "可以试试", "那我看看", "应该没问题", "好吧", "听你的", "可以可以"],
            "reject": ["不想跑了", "算了吧", "我不考虑", "暂时没兴趣", "不用了", "别说了", "算了"],
            "hesitate": ["我再想想", "有点担心", "会不会...", "真的吗？", "那要是...", "我再问问", "不确定"],
            "question": ["为啥啊？", "怎么弄？", "多少钱？", "真的假的？", "具体呢？", "多久啊？"],
            "acknowledge": ["嗯", "知道了", "行吧", "哦", "这样啊", "好吧", "晓得了"],
            "complain": ["太坑了", "又不公平", "老是扣钱", "根本没法干", "越来越难了"],
            "distracted": ["对了，那个...", "等会儿啊", "我先送完这单", "哎对了", "你说啥来着？"],
        }

    def pick_expression(self, intent: str) -> str | None:
        """Pick a diverse expression for the given intent.

        Returns:
            Selected expression or None if pool empty.
        """
        candidates = self.style_pool.get(intent, [])
        if not candidates:
            return None

        # Filter out recently used expressions
        available = [c for c in candidates if c not in self.recent_styles]
        if not available:
            available = candidates  # All used, reset

        import random
        chosen = random.choice(available)
        self.recent_styles.append(chosen)
        return chosen

    def should_regenerate(self, text: str) -> bool:
        """Check if text is semantically similar to recent expressions.

        Uses simple overlap-based heuristic.
        """
        if not self.recent_full_texts:
            return False

        # Check exact or near-exact match
        for recent in self.recent_full_texts:
            if text == recent:
                return True
            # Jaccard similarity on character bigrams
            similarity = self._char_bigram_similarity(text, recent)
            if similarity > 0.65:
                return True

        # Check high-frequency keyword repetition
        text_keywords = self._extract_keywords(text)
        for recent_kw in self.recent_keywords:
            if recent_kw in text_keywords:
                # If same keyword appears in 2+ recent texts, flag as repetitive
                count = sum(1 for kw in self.recent_keywords if kw == recent_kw)
                if count >= 2:
                    return True

        return False

    def track_usage(self, text: str, intent: str | None = None) -> None:
        """Record expression usage for future diversity checks."""
        self.recent_full_texts.append(text)
        if intent:
            self.recent_styles.append(intent)

        # Track keywords
        keywords = self._extract_keywords(text)
        for kw in keywords:
            self.recent_keywords.append(kw)

    def get_diversity_prompt_constraint(self) -> str:
        """Get prompt text for diversity constraints."""
        recent = list(self.recent_full_texts)
        constraint = """【表达多样性约束】
- 避免连续两轮使用同类表达（如连续两声"嗯"）。
- 避免高频词重复（如"好的""看看""再说"）。
"""
        if recent:
            constraint += f"- 你最近说过的：{'; '.join(recent)}。不要重复类似意思。\n"
        return constraint

    def _char_bigram_similarity(self, a: str, b: str) -> float:
        """Calculate character bigram Jaccard similarity."""
        if not a or not b:
            return 0.0

        def bigrams(s: str) -> set[str]:
            return {s[i:i + 2] for i in range(len(s) - 1)}

        bg_a = bigrams(a)
        bg_b = bigrams(b)
        intersection = len(bg_a & bg_b)
        union = len(bg_a | bg_b)
        return intersection / union if union > 0 else 0.0

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract simple keywords from text."""
        # Simple extraction: 2-3 char meaningful fragments
        keywords = []
        common = ["好的", "看看", "再说", "嗯嗯", "啊啊", "那个", "这个", "就是"]
        for kw in common:
            if kw in text:
                keywords.append(kw)
        return keywords
