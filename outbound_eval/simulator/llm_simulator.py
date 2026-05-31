"""LLM-based user simulator with Memory + State Machine + Diversity."""

from typing import Optional
from outbound_eval.agent.llm.base import LLMClient
from outbound_eval.scenarios.personas import UserPersona, PersonaType
from outbound_eval.scenarios.dialogue_strategies import (
    DialogueStrategy,
    STRATEGY_MAP,
)
from outbound_eval.scenarios.memory import ConversationMemory
from outbound_eval.scenarios.state_machine import UserStateMachine
from outbound_eval.scenarios.diversity import ResponseDiversityManager
from outbound_eval.events.injector import EventInjector


class UserSimulator:
    """Simulates user responses for evaluation with enhanced realism."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        event_injector: Optional[EventInjector] = None,
    ):
        """Initialize the simulator.

        Args:
            llm_client: LLM client for generating responses
            event_injector: Optional event injector
        """
        self.llm_client = llm_client
        self.event_injector = event_injector
        self.dialogue_history: list[dict] = []
        self.persona: Optional[UserPersona] = None
        self.strategy: Optional[DialogueStrategy] = None
        self.memory: Optional[ConversationMemory] = None
        self.state_machine: Optional[UserStateMachine] = None
        self.diversity: Optional[ResponseDiversityManager] = None

    def set_persona(self, persona: UserPersona) -> None:
        """Set the user persona and initialize subsystems.

        Args:
            persona: User persona to simulate
        """
        self.persona = persona
        self.dialogue_history = []

        # Get strategy for persona type
        strategy_class = STRATEGY_MAP.get(
            persona.persona_type.value,
            STRATEGY_MAP["cooperative"],
        )
        self.strategy = strategy_class()

        # Initialize memory
        self.memory = ConversationMemory()

        # Set emotion_level from persona's emotional_baseline
        baseline_map = {"愤怒": 5, "不耐烦": 4, "犹豫": 3, "急躁": 4, "平静": 2, "轻松": 1}
        self.memory.emotion_level = baseline_map.get(persona.emotional_baseline, 3)

        # Initialize state machine
        self.state_machine = UserStateMachine.from_persona_type(
            persona.persona_type.value,
            patience=persona.patience_level,
        )

        # Initialize diversity manager with persona's style pool
        pool = persona.response_style_pool
        if not pool:
            pool = persona.get_default_style_pool()
        self.diversity = ResponseDiversityManager(style_pool=pool)

    def generate_response(
        self, agent_message: str, current_turn: int = 0
    ) -> str:
        """Generate a user response.

        Args:
            agent_message: The agent's message
            current_turn: Current turn number

        Returns:
            Generated user response
        """
        if not self.persona or not self.strategy:
            raise ValueError("Persona not set. Call set_persona first.")

        # Update state machine based on agent message
        if self.state_machine:
            self.state_machine.transition(agent_message)

        # Check for injected event
        context = {
            "last_agent_message": agent_message,
            "dialogue_history": self.dialogue_history,
            "memory": self.memory,
            "state_machine": self.state_machine,
        }

        if self.event_injector:
            event = self.event_injector.get_next_event(current_turn, context)
            if event:
                self.event_injector.mark_injected(event.event_id)
                response = self.event_injector.process_event_response(
                    event, agent_message
                )
                self.dialogue_history.append(
                    {"role": "user", "content": response}
                )
                # Update memory with event response
                if self.memory:
                    self.memory.update(response, agent_message)
                return response

        # Pass enhanced context to strategy
        context["llm_client"] = self.llm_client
        response = self.strategy.generate_response(
            agent_message,
            self.persona,
            context,
            memory=self.memory,
            state_machine=self.state_machine,
            diversity=self.diversity,
        )

        # Update memory
        if self.memory:
            self.memory.update(response, agent_message)

        self.dialogue_history.append(
            {"role": "user", "content": response}
        )

        if self.event_injector:
            self.event_injector.advance_turn()

        return response

    def should_end_conversation(self) -> bool:
        """Check if conversation should end.

        Returns:
            True if conversation should end
        """
        if not self.strategy:
            return False

        context = {
            "dialogue_history": self.dialogue_history,
            "commitment_obtained": any(
                kw in turn["content"]
                for turn in self.dialogue_history
                for kw in ["可以", "好的", "没问题", "愿意"]
            ),
            "memory": self.memory,
            "state_machine": self.state_machine,
        }

        # Priority: state machine says end
        if self.state_machine and self.state_machine.should_end_conversation(
            self.dialogue_history
        ):
            return True

        return self.strategy.should_end_conversation(context)

    def get_dialogue_history(self) -> list[dict]:
        """Get the dialogue history.

        Returns:
            List of conversation turns
        """
        return self.dialogue_history.copy()

    def get_memory(self) -> Optional[ConversationMemory]:
        """Get conversation memory."""
        return self.memory

    def get_state_machine(self) -> Optional[UserStateMachine]:
        """Get state machine."""
        return self.state_machine

    def get_diversity_manager(self) -> Optional[ResponseDiversityManager]:
        """Get diversity manager."""
        return self.diversity

    def reset(self) -> None:
        """Reset the simulator."""
        self.dialogue_history = []
        self.persona = None
        self.strategy = None
        if self.memory:
            self.memory.reset()
            self.memory = None
        if self.state_machine:
            self.state_machine.reset()
            self.state_machine = None
        self.diversity = None
        if self.event_injector:
            self.event_injector.reset()
