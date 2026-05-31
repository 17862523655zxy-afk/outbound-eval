"""Conversation memory for user simulator."""

from pydantic import BaseModel, Field


class ConversationMemory(BaseModel):
    """Short-term memory module tracking conversation context."""

    mentioned_topics: list[str] = Field(default_factory=list)
    expressed_concerns: list[str] = Field(default_factory=list)
    emotion_level: int = Field(default=3, ge=1, le=5)
    commitment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    answered_questions: list[str] = Field(default_factory=list)
    user_commitments: list[str] = Field(default_factory=list)
    recent_expressions: list[str] = Field(default_factory=list)

    # Topic/concern extraction keywords
    _CONCERN_KEYWORDS: list[str] = [
        "担心", "怕", "罚款", "扣钱", "不安全", "太远", "太累", "没时间",
        "收入", "不稳定", "风险", "麻烦", "难", "贵", "不划算",
    ]
    _COMMITMENT_KEYWORDS: list[str] = [
        "可以", "好的", "没问题", "愿意", "行", "试试", "同意", "签", "答应",
    ]
    _QUESTION_KEYWORDS: list[str] = [
        "？", "什么", "怎么", "多少", "多久", "为什么", "哪", "吗",
    ]

    def update(self, user_message: str, agent_message: str) -> None:
        """Update memory based on the latest turn."""
        # Track recent expressions (keep last 3)
        self.recent_expressions.append(user_message)
        if len(self.recent_expressions) > 3:
            self.recent_expressions.pop(0)

        # Extract concerns from user message
        for kw in self._CONCERN_KEYWORDS:
            if kw in user_message:
                if kw not in self.expressed_concerns:
                    self.expressed_concerns.append(kw)

        # Track answered questions (agent answers user's previous questions)
        # Simple heuristic: if agent message is informative and user asked before
        if len(agent_message) > 10:
            self.answered_questions.append(agent_message[:50])
            if len(self.answered_questions) > 5:
                self.answered_questions.pop(0)

        # Track commitments from user
        for kw in self._COMMITMENT_KEYWORDS:
            if kw in user_message:
                self.commitment_score = min(1.0, self.commitment_score + 0.15)
                if user_message not in self.user_commitments:
                    self.user_commitments.append(user_message)

        # Update emotion level based on sentiment keywords
        negative = ["不", "没", "差", "烂", "坑", "骗", "气", "烦", "滚", "别"]
        positive = ["好", "行", "可以", "谢谢", "不错", "满意", "开心"]

        neg_count = sum(1 for w in negative if w in user_message)
        pos_count = sum(1 for w in positive if w in user_message)

        if neg_count > pos_count:
            self.emotion_level = min(5, self.emotion_level + 1)
        elif pos_count > neg_count:
            self.emotion_level = max(1, self.emotion_level - 1)

        # Extract mentioned topics (nouns and key phrases)
        # Simple heuristic: extract 2-4 character phrases
        self._extract_topics(user_message)

    def _extract_topics(self, message: str) -> None:
        """Extract potential topics from message."""
        topic_indicators = [
            "合同", "配送", "单量", "罚款", "收入", "时间", "距离",
            "保险", "补贴", "奖励", "申诉", "站长", "平台", "换",
        ]
        for indicator in topic_indicators:
            if indicator in message and indicator not in self.mentioned_topics:
                self.mentioned_topics.append(indicator)

    def get_memory_context(self) -> str:
        """Generate memory context text for prompts."""
        parts = []

        if self.mentioned_topics:
            parts.append(f"你已提及的话题：{', '.join(self.mentioned_topics[-5:])}")
        if self.expressed_concerns:
            parts.append(f"你已表达的顾虑：{', '.join(self.expressed_concerns[-3:])}")
        if self.answered_questions:
            parts.append(f"站长已回答的问题：{len(self.answered_questions)}个")
        if self.user_commitments:
            parts.append(f"你已做出的承诺：{', '.join(self.user_commitments[-2:])}")

        parts.append(f"当前情绪等级：{self.emotion_level}/5")
        parts.append(f"承诺度：{self.commitment_score:.0%}")

        return "\n".join(parts)

    def is_recently_expressed(self, text: str) -> bool:
        """Check if similar content was expressed in recent 3 turns."""
        if not self.recent_expressions:
            return False

        # Simple overlap check: if any recent expression contains
        # significant keywords from the new text
        text_keywords = set(text)
        for recent in self.recent_expressions:
            recent_keywords = set(recent)
            overlap = len(text_keywords & recent_keywords)
            if overlap >= min(len(text_keywords), len(recent_keywords)) * 0.6:
                return True
        return False

    def is_question_answered(self, question: str) -> bool:
        """Check if a question has already been answered by agent."""
        for answered in self.answered_questions:
            # Simple keyword overlap
            if any(kw in answered for kw in question[:10]):
                return True
        return False

    def reset(self) -> None:
        """Reset all memory."""
        self.mentioned_topics = []
        self.expressed_concerns = []
        self.emotion_level = 3
        self.commitment_score = 0.0
        self.answered_questions = []
        self.user_commitments = []
        self.recent_expressions = []
