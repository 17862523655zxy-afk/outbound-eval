"""User behavior state machine for realistic state transitions."""

from enum import Enum
from pydantic import BaseModel, Field


class UserState(str, Enum):
    """User emotional/commitment states."""

    HARD_REJECT = "hard_reject"
    SOFT_REJECT = "soft_reject"
    CONSIDERING = "considering"
    INTERESTED = "interested"
    COMMITTED = "committed"


# Default initial state by persona type
DEFAULT_INITIAL_STATES: dict[str, UserState] = {
    "cooperative": UserState.INTERESTED,
    "indecisive": UserState.CONSIDERING,
    "rejection": UserState.HARD_REJECT,
    "emotional": UserState.HARD_REJECT,
    "off_topic": UserState.CONSIDERING,
}


# State prompt fragments for LLM
STATE_PROMPT_FRAGMENTS: dict[UserState, str] = {
    UserState.HARD_REJECT: (
        "你目前坚决拒绝，语气硬，带点不耐烦。"
        "如果站长能解决你的核心顾虑（收入、罚款、时间），你可能软化。"
        "不要突然变好，保持抵触情绪。"
    ),
    UserState.SOFT_REJECT: (
        "你已经开始犹豫，没那么坚决了，但还在观望。"
        "如果站长继续解释好处、打消顾虑，你可能进一步考虑。"
        "可以偶尔问一两个具体问题。"
    ),
    UserState.CONSIDERING: (
        "你正在认真考虑，但还有具体顾虑没被打消。"
        "如果站长能明确回答你的问题、提供支持，你可能会有兴趣。"
        "可以表达一些试探性的同意。"
    ),
    UserState.INTERESTED: (
        "你基本同意了，但还想确认一些细节。"
        "如果站长能明确回答细节、给出口头确认，你可能做出承诺。"
        "语气开始积极，但还有一丝保留。"
    ),
    UserState.COMMITTED: (
        "你已经同意了，可以给出明确承诺或同意操作。"
        "语气积极，不再提反对意见。"
        "可以说'行吧，那就这样'、'我配合'。"
    ),
}


class UserStateMachine(BaseModel):
    """State machine tracking user commitment progression."""

    current_state: UserState = Field(default=UserState.CONSIDERING)
    transition_log: list[dict] = Field(default_factory=list)
    _same_direction_count: int = 0
    _last_direction: str = ""
    MAX_SAME_DIRECTION: int = 3
    patience_level: int = Field(default=3, ge=1, le=5)

    @classmethod
    def from_persona_type(cls, persona_type: str, patience: int = 3) -> "UserStateMachine":
        """Create state machine with default initial state for persona."""
        initial = DEFAULT_INITIAL_STATES.get(persona_type, UserState.CONSIDERING)
        return cls(current_state=initial, patience_level=patience)

    def transition(self, agent_message: str) -> UserState:
        """Transition state based on agent message content."""
        old_state = self.current_state
        new_state = self._compute_transition(agent_message)

        if new_state != old_state:
            # Track direction to prevent oscillation
            direction = self._get_direction(old_state, new_state)
            if direction == self._last_direction:
                self._same_direction_count += 1
            else:
                self._same_direction_count = 1
                self._last_direction = direction

            # Prevent oscillation
            if self._same_direction_count > self.MAX_SAME_DIRECTION:
                # Stay in current state
                return old_state

            self.current_state = new_state
            self.transition_log.append({
                "from": old_state.value,
                "to": new_state.value,
                "trigger": agent_message[:80],
            })

        return self.current_state

    def _compute_transition(self, agent_message: str) -> UserState:
        """Compute new state from agent message keywords."""
        msg = agent_message.lower()
        current = self.current_state

        # Positive signals (agent is helpful/solving problems)
        positive_signals = [
            "解决", "帮忙", "支持", "补贴", "奖励", "好处", "放心",
            "保证", "承诺", "具体", "详细", "解释", "说明", "优势",
            "收入", "赚钱", "稳定", "保险", "保障", "安全",
        ]
        # Negative signals (agent is pushy/unclear)
        negative_signals = [
            "必须", "一定", "赶紧", "快点", "催促", "强制", "要求",
            "不懂", "不清楚", "再说", "自己看", "不知道",
        ]

        pos_score = sum(1 for s in positive_signals if s in msg)
        neg_score = sum(1 for s in negative_signals if s in msg)

        # State transition logic
        if current == UserState.HARD_REJECT:
            if pos_score >= 2 and neg_score == 0:
                return UserState.SOFT_REJECT
        elif current == UserState.SOFT_REJECT:
            if pos_score >= 2 and neg_score == 0:
                return UserState.CONSIDERING
            if neg_score >= 2:
                return UserState.HARD_REJECT
        elif current == UserState.CONSIDERING:
            if pos_score >= 3 and neg_score == 0:
                return UserState.INTERESTED
            if neg_score >= 2:
                return UserState.SOFT_REJECT
        elif current == UserState.INTERESTED:
            if pos_score >= 1 and neg_score == 0:
                return UserState.COMMITTED
            if neg_score >= 1:
                return UserState.CONSIDERING
        elif current == UserState.COMMITTED:
            if neg_score >= 2:
                return UserState.INTERESTED

        return current

    def _get_direction(self, old: UserState, new: UserState) -> str:
        """Get transition direction: 'positive' or 'negative'."""
        order = [UserState.HARD_REJECT, UserState.SOFT_REJECT,
                 UserState.CONSIDERING, UserState.INTERESTED, UserState.COMMITTED]
        old_idx = order.index(old)
        new_idx = order.index(new)
        if new_idx > old_idx:
            return "positive"
        elif new_idx < old_idx:
            return "negative"
        return "same"

    def get_prompt_fragment(self) -> str:
        """Get prompt fragment for current state."""
        return STATE_PROMPT_FRAGMENTS.get(
            self.current_state,
            "保持你当前的态度回复。"
        )

    def should_end_conversation(self, dialogue_history: list[dict]) -> bool:
        """Check if conversation should naturally end."""
        max_turns = self.patience_level * 5  # patience=5→25轮, patience=1→5轮
        if self.current_state == UserState.COMMITTED or len(dialogue_history) >= max_turns:
            return True
        return False

    def get_transition_log(self) -> list[dict]:
        """Get transition history."""
        return self.transition_log.copy()

    def reset(self) -> None:
        """Reset state machine."""
        self.current_state = UserState.CONSIDERING
        self.transition_log = []
        self._same_direction_count = 0
        self._last_direction = ""
