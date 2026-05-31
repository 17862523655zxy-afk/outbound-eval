"""Outbound agent implementation."""

from typing import Optional
from outbound_eval.agent.llm.base import LLMClient
from outbound_eval.agent.llm.prompt_builder import PromptBuilder
from outbound_eval.agent.skills.base import SkillScript
from outbound_eval.agent.skills.loader import SkillLoader
from outbound_eval.agent.skills.constraints import ConstraintEngine
from outbound_eval.agent.dialog_manager import DialogManager
from outbound_eval.agent.flow_engine import FlowEngine
from outbound_eval.agent.state_manager import StateManager


class OutboundAgent:
    """Main outbound call agent."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        skill_loader: Optional[SkillLoader] = None,
    ):
        """Initialize the agent.

        Args:
            llm_client: LLM client for generating responses
            skill_loader: Loader for skill scripts
        """
        from outbound_eval.agent.llm.openai_client import OpenAIClient

        self.llm_client = llm_client or OpenAIClient()
        self.skill_loader = skill_loader or SkillLoader()

        # Core components
        self.dialog_manager = DialogManager()
        self.flow_engine = FlowEngine()
        self.state_manager = StateManager()
        self.prompt_builder = PromptBuilder()

        # Current skill
        self.current_skill: Optional[SkillScript] = None
        self.constraint_engine: Optional[ConstraintEngine] = None

    def load_skill(self, skill_name: str, **variables) -> None:
        """Load a skill script.

        Args:
            skill_name: Name of the skill to load
            **variables: Variable values for the skill
        """
        self.current_skill = self.skill_loader.load(skill_name)
        self.current_skill = self.current_skill.fill_variables(**variables)

        # Setup constraint engine
        self.constraint_engine = ConstraintEngine(
            self.current_skill.constraints
        )

        # Build system prompt
        self.prompt_builder.reset()
        self.prompt_builder.with_role(self.current_skill.role)

        # Add constraints
        constraints = [
            f"每次回复控制在{self.current_skill.constraints.get('max_response_length', 50)}个字以内",
            "你正在打电话，语气要口语化、自然，像真人说话",
            "可以使用'嗯'、'啊'、'吧'、'呢'等语气词",
            "不要像客服机器人一样完美，可以有小停顿",
            "不要说书面语，要说大白话",
            "可以重复强调重点，但不要太机械",
            "如果对方犹豫，要耐心引导",
            "如果对方生气，要先安抚情绪",
            "避免重复同样的句子",
        ]
        self.prompt_builder.with_constraints(constraints)

        # Add flow
        if self.current_skill.flow:
            flow_text = "\n".join(
                f"Step {f['step']}: {f['description']}"
                for f in self.current_skill.flow
            )
            self.prompt_builder.with_flow(flow_text)
            # Ensure critical steps are emphasized
            self.prompt_builder.with_constraints([
                "重要：必须在对话中提醒骑手注意安全",
                "重要：必须说明排名规则（减少拒单取消可以提高排名）",
            ])

        # Add FAQ
        if self.current_skill.faq:
            self.prompt_builder.with_faq(self.current_skill.faq)

    def start_conversation(self) -> str:
        """Start a new conversation.

        Returns:
            The opening line
        """
        if not self.current_skill:
            raise ValueError("No skill loaded. Call load_skill first.")

        self.dialog_manager.reset()
        self.state_manager.reset()
        self.flow_engine.reset(self.current_skill.flow)

        opening = self.current_skill.opening_line

        # Record the opening
        self.dialog_manager.add_turn("agent", opening)

        # Update state
        self.state_manager.update_state("conversation_started", True)
        self.state_manager.update_state("current_step", "confirm_identity")

        return opening

    def respond(self, user_message: str) -> str:
        """Generate a response to user message.

        Args:
            user_message: The user's message

        Returns:
            The agent's response
        """
        if not self.current_skill:
            raise ValueError("No skill loaded.")

        # Add user message to history
        self.dialog_manager.add_turn("user", user_message)

        # Build prompt
        system_prompt = self.prompt_builder.build_system()
        user_prompt = self.prompt_builder.build_user_message(
            user_message,
            self.dialog_manager.get_history(),
        )

        # Generate response
        response = self.llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )

        # Apply constraints
        if self.constraint_engine:
            is_valid, errors = self.constraint_engine.check_all(response)
            if not is_valid:
                # Truncate if too long
                response = self.constraint_engine.truncate_response(response)

        # Add to history
        self.dialog_manager.add_turn("agent", response)

        # Update state based on response
        self._update_state_from_response(response, user_message)

        # Update flow
        self.flow_engine.record_turn(user_message, response)

        return response

    def _update_state_from_response(self, agent_response: str, user_message: str) -> None:
        """Update agent state based on responses.

        Args:
            agent_response: The agent's response
            user_message: The user's message
        """
        # Track user intent
        if any(kw in user_message for kw in ["是", "对", "没错"]):
            self.state_manager.update_state("identity_confirmed", True)

        if any(kw in user_message for kw in ["可以", "好的", "没问题", "愿意"]):
            self.state_manager.update_state("commitment_obtained", True)
            self.state_manager.update_state("user_intent", "confirm")

        if any(kw in user_message for kw in ["不送了", "不做了", "退出", "不想"]):
            self.state_manager.update_state("user_intent", "reject")
            self.state_manager.update_state("commitment_obtained", False)

        if "安全" in agent_response:
            self.state_manager.update_state("safety_reminded", True)

    def should_end_conversation(self) -> bool:
        """Check if conversation should end.

        Returns:
            True if conversation should end
        """
        state = self.state_manager.get_state()

        # End if commitment obtained and key info delivered
        if state.get("commitment_obtained") and state.get("safety_reminded"):
            return True

        # End if user explicitly refuses
        if state.get("user_intent") == "reject" and state.get("retention_attempts", 0) >= 2:
            return True

        return False

    def get_conversation_history(self) -> list[dict]:
        """Get the conversation history.

        Returns:
            List of conversation turns
        """
        return self.dialog_manager.get_history()

    def get_current_state(self) -> dict:
        """Get the current agent state.

        Returns:
            Current state dict
        """
        return self.state_manager.get_state()

    def get_flow_progress(self) -> dict:
        """Get flow progress.

        Returns:
            Flow progress dict
        """
        return self.flow_engine.get_progress()