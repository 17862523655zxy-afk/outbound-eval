"""Event injection for complex scenarios."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events that can be injected."""

    INTERRUPTION = "interruption"
    REJECTION = "rejection"
    EMOTIONAL = "emotional"
    INFO_CHANGE = "info_change"
    OFF_TOPIC = "off_topic"
    BACKGROUND_NOISE = "background_noise"
    DROPPED_CALL = "dropped_call"
    # Phone-specific events
    NETWORK_ISSUE = "network_issue"
    MULTITASKING = "multitasking"


class InjectionTrigger(str, Enum):
    """Trigger types for event injection."""

    TURN_BASED = "turn_based"
    INTENT_BASED = "intent_based"
    TIME_BASED = "time_based"
    RANDOM = "random"


class InjectedEvent(BaseModel):
    """Definition of an event to be injected."""

    event_id: str = Field(description="Unique event ID")
    event_type: EventType = Field(description="Type of event")
    trigger: InjectionTrigger = Field(
        default=InjectionTrigger.TURN_BASED, description="Trigger type"
    )

    # Trigger conditions
    inject_at_turn: Optional[int] = Field(
        default=None, description="Turn number to inject at (turn_based)"
    )
    trigger_keywords: list[str] = Field(
        default_factory=list, description="Keywords to trigger on (intent_based)"
    )

    # Event payload
    payload: dict = Field(
        default_factory=dict, description="Event payload data"
    )

    # Weight for random selection
    weight: float = Field(default=1.0, description="Event weight")


class EventInjector:
    """Injects events into conversations to create complex scenarios."""

    # Default phone scenario events with probabilities
    PHONE_EVENT_TEMPLATES: list[InjectedEvent] = [
        InjectedEvent(
            event_id="phone_network_issue_1",
            event_type=EventType.NETWORK_ISSUE,
            trigger=InjectionTrigger.RANDOM,
            weight=0.08,
            payload={"response_text": "喂？刚没听清，你再说一遍。"},
        ),
        InjectedEvent(
            event_id="phone_network_issue_2",
            event_type=EventType.NETWORK_ISSUE,
            trigger=InjectionTrigger.RANDOM,
            weight=0.05,
            payload={"response_text": "信号有点差，断断续续的。"},
        ),
        InjectedEvent(
            event_id="phone_interruption_1",
            event_type=EventType.INTERRUPTION,
            trigger=InjectionTrigger.RANDOM,
            weight=0.05,
            payload={"response_text": "等下，我在送单，马上到。"},
        ),
        InjectedEvent(
            event_id="phone_interruption_2",
            event_type=EventType.INTERRUPTION,
            trigger=InjectionTrigger.RANDOM,
            weight=0.04,
            payload={"response_text": "（喇叭声）等会儿啊，我先回个消息。"},
        ),
        InjectedEvent(
            event_id="phone_multitasking_1",
            event_type=EventType.MULTITASKING,
            trigger=InjectionTrigger.RANDOM,
            weight=0.06,
            payload={"response_text": "你等下啊，我快到了，先不说了..."},
        ),
        InjectedEvent(
            event_id="phone_multitasking_2",
            event_type=EventType.MULTITASKING,
            trigger=InjectionTrigger.RANDOM,
            weight=0.05,
            payload={"response_text": "嗯...（喘气）我刚爬完楼。"},
        ),
        InjectedEvent(
            event_id="phone_background_noise_1",
            event_type=EventType.BACKGROUND_NOISE,
            trigger=InjectionTrigger.RANDOM,
            weight=0.03,
            payload={"response_text": "（风声）你大声点，外面太吵了。"},
        ),
    ]

    def __init__(self, event_library: Optional[list[InjectedEvent]] = None):
        """Initialize the injector.

        Args:
            event_library: List of available events. If None, uses default phone events.
        """
        self.event_library = event_library or self.PHONE_EVENT_TEMPLATES.copy()
        self.injected_event_ids: set[str] = set()
        self.turn_count = 0
        self._last_event_type: Optional[str] = None

    def should_inject(
        self, turn: int, context: dict, event: InjectedEvent
    ) -> bool:
        """Check if an event should be injected.

        Args:
            turn: Current turn number
            context: Conversation context
            event: Event to check

        Returns:
            True if event should be injected
        """
        if event.event_id in self.injected_event_ids:
            return False

        if event.trigger == InjectionTrigger.TURN_BASED:
            return turn == event.inject_at_turn

        elif event.trigger == InjectionTrigger.INTENT_BASED:
            last_agent_message = context.get("last_agent_message", "")
            return any(kw in last_agent_message for kw in event.trigger_keywords)

        elif event.trigger == InjectionTrigger.RANDOM:
            import random
            return random.random() < event.weight * 0.3

        return False

    def get_next_event(
        self, turn: int, context: dict
    ) -> Optional[InjectedEvent]:
        """Get the next event to inject based on conditions.

        Args:
            turn: Current turn number
            context: Conversation context

        Returns:
            Event to inject or None
        """
        # Conflict prevention: skip if an event was injected in the last 2 turns
        recent_injected = [
            eid for eid in self.injected_event_ids
            if eid.startswith("phone_")
        ]
        if len(recent_injected) >= 1 and self.turn_count <= 2:
            return None

        for event in self.event_library:
            if self.should_inject(turn, context, event):
                # Skip if same type was recently injected
                if event.event_type.value == self._last_event_type:
                    continue
                self._last_event_type = event.event_type.value
                return event

        return None

    def mark_injected(self, event_id: str) -> None:
        """Mark an event as injected.

        Args:
            event_id: Event ID that was injected
        """
        self.injected_event_ids.add(event_id)

    def process_event_response(
        self, event: InjectedEvent, agent_response: str
    ) -> str:
        """Process the user's response to an injected event.

        Args:
            event: The injected event
            agent_response: Agent's response

        Returns:
            User's response to the event
        """
        event_type = event.event_type
        payload = event.payload

        if event_type == EventType.INTERRUPTION:
            return payload.get("response_text", "等等，你先把话说完。")

        elif event_type == EventType.REJECTION:
            return payload.get("rejection_text", "我不想做了。")

        elif event_type == EventType.EMOTIONAL:
            return payload.get("emotional_text", "我太生气了！")

        elif event_type == EventType.OFF_TOPIC:
            return payload.get("off_topic_text", "对了，你们那边几点下班？")

        elif event_type == EventType.INFO_CHANGE:
            return payload.get("change_text", "等等，我之前说的好像有问题...")

        elif event_type == EventType.BACKGROUND_NOISE:
            return payload.get("response_text", "[背景噪音] ...")

        elif event_type == EventType.DROPPED_CALL:
            return "[通话中断] ..."

        elif event_type == EventType.NETWORK_ISSUE:
            return payload.get("response_text", "喂？刚没听清。")

        elif event_type == EventType.MULTITASKING:
            return payload.get("response_text", "等下，我在忙。")

        return ""

    def advance_turn(self) -> None:
        """Advance the turn counter."""
        self.turn_count += 1

    def reset(self) -> None:
        """Reset the injector state."""
        self.injected_event_ids = set()
        self.turn_count = 0
        self._last_event_type = None

    def load_from_config(self, events_config: list[dict]) -> None:
        """Load events from config dict.

        Args:
            events_config: List of event config dicts
        """
        self.event_library = [InjectedEvent(**config) for config in events_config]

    def get_triggered_events_log(self) -> list[dict]:
        """Get log of triggered events for analysis.

        Returns:
            List of triggered event records.
        """
        log = []
        for eid in self.injected_event_ids:
            for event in self.event_library:
                if event.event_id == eid:
                    log.append({
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "turn": self.turn_count,
                    })
                    break
        return log