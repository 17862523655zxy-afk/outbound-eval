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

        # Update state from user message BEFORE generating response
        self._update_state_from_user_message(user_message)

        # Check if we should force-end the conversation
        state = self.state_manager.get_state()
        retention = state.get("retention_attempts", 0)
        user_intent = state.get("user_intent", "")
        history_len = len(self.dialog_manager.get_history())

        # Force ending: user rejected twice or conversation too long
        should_force_end = (
            (user_intent == "reject" and retention >= 2)
            or history_len >= 16
        )

        if should_force_end:
            # Return a forced closing message, bypass LLM
            closing = self._generate_closing_message(state)
            self.dialog_manager.add_turn("agent", closing)
            self.flow_engine.record_turn(user_message, closing)
            # Mark safety reminded if not already
            if "安全" in closing:
                self.state_manager.update_state("safety_reminded", True)
            return closing

        # Build prompt with dynamic step guidance
        system_prompt = self._build_system_prompt_with_guidance()
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

        # Update state based on agent response
        self._update_state_from_agent_response(response)

        # Update flow
        self.flow_engine.record_turn(user_message, response)

        # Check flow step completion and advance
        current_step = self.flow_engine.get_current_step()
        if current_step:
            step_id = current_step["id"]
            is_complete = self.flow_engine.check_step_completion(
                step_id, user_message, response
            )
            if is_complete:
                self.flow_engine.advance_step()

        return response

    def _generate_closing_message(self, state: dict) -> str:
        """Generate a polite closing message based on conversation outcome."""
        if state.get("commitment_obtained"):
            return "好的，那就先这样。你跑单的时候注意安全，有问题随时联系我。"
        else:
            return "行，那不打扰你了。你注意安全，要是改主意了随时联系我。"

    def _build_system_prompt_with_guidance(self) -> str:
        """Build system prompt with dynamic step guidance and end-game hints."""
        base_prompt = self.prompt_builder.build_system()

        # Determine current step and add guidance
        current_step = self.flow_engine.get_current_step()
        guidance_parts = []

        if current_step:
            step_desc = current_step.get("description", "")
            step_id = current_step.get("id", "")
            guidance_parts.append(f"\n【当前流程步骤】{step_desc}")

            # Add keyword hints for the current step
            keywords = current_step.get("expected_keywords", {})
            if isinstance(keywords, dict):
                req = keywords.get("required", [])
                if req:
                    guidance_parts.append(f"【必须提及】{'、'.join(req)}")
                confirm = keywords.get("confirm", [])
                if confirm:
                    guidance_parts.append(f"【需确认】{'、'.join(confirm)}")

        # End-game guidance based on conversation state
        state = self.state_manager.get_state()
        user_intent = state.get("user_intent", "")
        retention = state.get("retention_attempts", 0)

        if user_intent == "reject" and retention >= 2:
            guidance_parts.append("\n【结束指引】用户已多次拒绝，本次是最后回复。请礼貌结束通话，结束前务必提醒'注意安全'，并说'那不打扰你了，有问题随时联系我'。")
        elif user_intent == "reject" and retention == 1:
            guidance_parts.append("\n【挽留指引】用户有拒绝意向，请尝试挽留一次：说明跑单的好处、平台支持、或者柔性劝导。不要强行推销。")
        elif state.get("commitment_obtained") and not state.get("safety_reminded"):
            guidance_parts.append("\n【收尾指引】用户已同意，结束前务必提醒'注意安全'，并说'好的，那就这样，有问题随时联系'。")
        elif state.get("commitment_obtained") and state.get("safety_reminded"):
            guidance_parts.append("\n【收尾指引】用户已同意且安全已提醒，礼貌结束通话。")

        if guidance_parts:
            base_prompt += "\n" + "\n".join(guidance_parts)

        return base_prompt

    def _update_state_from_user_message(self, user_message: str) -> None:
        """Update agent state based on user message (called once per turn)."""
        # Track user intent
        if any(kw in user_message for kw in ["是", "对", "没错"]):
            self.state_manager.update_state("identity_confirmed", True)

        if any(kw in user_message for kw in ["可以", "好的", "没问题", "愿意", "行", "同意", "答应"]):
            self.state_manager.update_state("commitment_obtained", True)
            self.state_manager.update_state("user_intent", "confirm")
            return  # Positive intent clears rejection

        reject_keywords = [
            "不送了", "不做了", "退出", "不想", "不跑", "不干", "不签",
            "拒绝", "不同意", "不行", "算了吧", "别打了", "不干了",
            "不答应", "不愿意", "没兴趣", "别找我", "别联系", "烦不烦",
            "谁干谁亏", "谁受得了", "谁顶得住", "我图啥", "给平台打工",
            "说得轻巧", "还试水呢", "冷笑",
            "有啥用", "倒贴", "坑人", "套路", "忽悠",
        ]
        strong_negative = [
            "骗", "坑", "剥削", "欺负", "不公平", "想怎么罚就怎么罚",
            "不平等", "压迫", "扣钱", "罚款", "白干", "压榨",
        ]
        dismissive = [
            "别说了", "别讲了", "懒得说", "不想听", "不想谈",
            "没意义", "没用", "算了", "拉倒", "别提了",
        ]

        is_reject = any(kw in user_message for kw in reject_keywords)
        is_strongly_negative = any(kw in user_message for kw in strong_negative)
        is_dismissive = any(kw in user_message for kw in dismissive)

        if is_reject or is_strongly_negative or is_dismissive:
            prev_intent = self.state_manager.get_state().get("user_intent")
            self.state_manager.update_state("user_intent", "reject")
            self.state_manager.update_state("commitment_obtained", False)
            # Count retention attempts (cap at 2)
            if prev_intent == "reject":
                attempts = self.state_manager.get_state().get("retention_attempts", 0)
                if attempts < 2:
                    self.state_manager.update_state("retention_attempts", attempts + 1)
            else:
                self.state_manager.update_state("retention_attempts", 1)

    def _update_state_from_agent_response(self, agent_response: str) -> None:
        """Update agent state based on agent response (called once per turn)."""
        if "安全" in agent_response:
            self.state_manager.update_state("safety_reminded", True)

        if "排名" in agent_response and ("拒单" in agent_response or "取消" in agent_response):
            self.state_manager.update_state("ranking_explained", True)

    def should_end_conversation(self) -> bool:
        """Check if conversation should end.

        Returns:
            True if conversation should end
        """
        state = self.state_manager.get_state()

        # End if commitment obtained and key info delivered
        if state.get("commitment_obtained") and state.get("safety_reminded"):
            return True

        # End if user explicitly refuses after 2 retention attempts
        if state.get("user_intent") == "reject" and state.get("retention_attempts", 0) >= 2:
            return True

        # End if user rejects and agent has already said goodbye/ending words
        if state.get("user_intent") == "reject":
            history = self.dialog_manager.get_history()
            last_agent_msg = ""
            for turn in reversed(history):
                if turn.get("role") == "agent":
                    last_agent_msg = turn.get("content", "")
                    break
            if any(kw in last_agent_msg for kw in ["不打扰", "先这样", "挂了", "再见", "随时联系"]):
                return True

        # Fallback: too long
        if len(self.dialog_manager.get_history()) >= 18:
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