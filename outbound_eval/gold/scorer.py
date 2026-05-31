"""Gold conversation scorer."""

from typing import Optional
from outbound_eval.gold.conversation import GoldConversation, DialogTurn
from outbound_eval.gold.loader import GoldConversationLoader


class GoldComparisonResult:
    """Result of comparing with gold conversation."""

    def __init__(
        self,
        overall_similarity: float,
        turn_scores: list[float],
        flagged_anomalies: list[dict],
    ):
        self.overall_similarity = overall_similarity
        self.turn_scores = turn_scores
        self.flagged_anomalies = flagged_anomalies


class GoldConversationScorer:
    """Scores conversations by comparing with gold standards."""

    def __init__(
        self,
        gold_loader: Optional[GoldConversationLoader] = None,
        similarity_threshold: float = 0.7,
    ):
        """Initialize the scorer.

        Args:
            gold_loader: Loader for gold conversations
            similarity_threshold: Threshold for flagging anomalies
        """
        self.gold_loader = gold_loader or GoldConversationLoader()
        self.similarity_threshold = similarity_threshold

    def compare_with_gold(
        self,
        actual_dialogue: list[dict],
        task_id: str,
        persona_type: str = "cooperative",
    ) -> GoldComparisonResult:
        """Compare actual dialogue with gold standard.

        Args:
            actual_dialogue: List of actual conversation turns
            task_id: Task ID
            persona_type: User persona type

        Returns:
            Comparison result
        """
        # Load gold conversations
        gold_convs = self.gold_loader.load_for_task(task_id, persona_type)
        if not gold_convs:
            return GoldComparisonResult(
                overall_similarity=0.5,
                turn_scores=[0.5] * len(actual_dialogue),
                flagged_anomalies=[],
            )

        # Use the best matching gold conversation
        best_gold = self._find_best_match(actual_dialogue, gold_convs)

        # Calculate turn-by-turn similarity
        turn_scores = self._calculate_turn_scores(actual_dialogue, best_gold)

        # Find anomalies
        anomalies = self._find_anomalies(actual_dialogue, best_gold)

        # Overall similarity
        overall = sum(turn_scores) / len(turn_scores) if turn_scores else 0.5

        return GoldComparisonResult(
            overall_similarity=overall,
            turn_scores=turn_scores,
            flagged_anomalies=anomalies,
        )

    def _find_best_match(
        self,
        actual: list[dict],
        gold_convs: list[GoldConversation],
    ) -> GoldConversation:
        """Find the best matching gold conversation.

        Args:
            actual: Actual dialogue
            gold_convs: List of gold conversations

        Returns:
            Best matching GoldConversation
        """
        best = gold_convs[0]
        best_score = 0.0

        for gold in gold_convs:
            score = self._calculate_similarity(actual, gold)
            if score > best_score:
                best_score = score
                best = gold

        return best

    def _calculate_similarity(
        self, actual: list[dict], gold: GoldConversation
    ) -> float:
        """Calculate overall similarity with gold conversation.

        Args:
            actual: Actual dialogue
            gold: Gold conversation

        Returns:
            Similarity score 0.0-1.0
        """
        if not actual or not gold.turns:
            return 0.0

        # Compare lengths
        length_ratio = min(len(actual), len(gold.turns)) / max(
            len(actual), len(gold.turns)
        )

        # Compare keywords
        agent_actual = [t["content"] for t in actual if t["role"] == "agent"]
        agent_gold = [t.content for t in gold.get_agent_turns()]

        keyword_matches = 0
        total_keywords = 0

        for gold_text in agent_gold:
            keywords = self._extract_keywords(gold_text)
            total_keywords += len(keywords)
            for kw in keywords:
                if any(kw in text for text in agent_actual):
                    keyword_matches += 1

        keyword_score = (
            keyword_matches / total_keywords if total_keywords > 0 else 0.5
        )

        return (length_ratio * 0.3 + keyword_score * 0.7)

    def _calculate_turn_scores(
        self, actual: list[dict], gold: GoldConversation
    ) -> list[float]:
        """Calculate similarity score for each turn.

        Args:
            actual: Actual dialogue
            gold: Gold conversation

        Returns:
            List of turn scores
        """
        scores = []

        for i, turn in enumerate(actual):
            if i < len(gold.turns):
                gold_turn = gold.turns[i]
                score = self._turn_similarity(turn, gold_turn)
            else:
                score = 0.5  # No gold reference

            scores.append(score)

        return scores

    def _turn_similarity(self, actual: dict, gold: DialogTurn) -> float:
        """Calculate similarity for a single turn.

        Args:
            actual: Actual turn
            gold: Gold turn

        Returns:
            Similarity score 0.0-1.0
        """
        if actual["role"] != gold.speaker:
            return 0.0

        actual_text = actual["content"]
        gold_text = gold.content

        # Simple word overlap
        actual_words = set(actual_text)
        gold_words = set(gold_text)

        if not gold_words:
            return 1.0

        overlap = len(actual_words & gold_words)
        return overlap / len(gold_words)

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract key content words.

        Args:
            text: Text to extract from

        Returns:
            Set of keywords
        """
        # Simple extraction: words > 2 chars
        words = []
        current = ""

        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                # Chinese character
                current += char
                if len(current) >= 2:
                    words.append(current)
                    current = ""
            else:
                if current:
                    words.append(current)
                    current = ""
                if char.isalnum():
                    current += char

        if current:
            words.append(current)

        # Filter and return
        return {w for w in words if len(w) >= 2}

    def _find_anomalies(
        self, actual: list[dict], gold: GoldConversation
    ) -> list[dict]:
        """Find anomalous turns compared to gold.

        Args:
            actual: Actual dialogue
            gold: Gold conversation

        Returns:
            List of anomaly dicts
        """
        anomalies = []

        for i, turn in enumerate(actual):
            if i < len(gold.turns):
                score = self._turn_similarity(turn, gold.turns[i])
                if score < self.similarity_threshold:
                    anomalies.append(
                        {
                            "turn": i,
                            "role": turn["role"],
                            "actual": turn["content"],
                            "expected": gold.turns[i].content,
                            "similarity": score,
                        }
                    )

        return anomalies