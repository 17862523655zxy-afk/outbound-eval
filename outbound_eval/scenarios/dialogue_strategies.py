"""Dialogue strategies for user simulation with Memory + State Machine + Diversity."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from outbound_eval.scenarios.personas import UserPersona
    from outbound_eval.scenarios.memory import ConversationMemory
    from outbound_eval.scenarios.state_machine import UserStateMachine
    from outbound_eval.scenarios.diversity import ResponseDiversityManager


class DialogueStrategy(ABC):
    """Base class for dialogue strategies."""

    @abstractmethod
    def generate_response(
        self,
        agent_message: str,
        persona: "UserPersona",
        context: dict,
        memory: Optional["ConversationMemory"] = None,
        state_machine: Optional["UserStateMachine"] = None,
        diversity: Optional["ResponseDiversityManager"] = None,
    ) -> str:
        """Generate a response based on strategy."""
        pass

    @abstractmethod
    def should_end_conversation(self, context: dict) -> bool:
        """Check if conversation should end."""
        pass

    def _build_system_prompt(
        self,
        persona: "UserPersona",
        memory: Optional["ConversationMemory"] = None,
        state_machine: Optional["UserStateMachine"] = None,
    ) -> str:
        """Build a human-like system prompt for the persona."""
        parts = [f"你是{persona.name}。你正在接一个外卖站长的电话。"]

        # Enhanced persona profile
        if persona.motivation:
            parts.append(f"你的核心动机：{persona.motivation}")
        if persona.current_status:
            parts.append(f"你当前的状态：{persona.current_status}")
        if persona.concerns:
            parts.append(f"你的顾虑：{', '.join(persona.concerns)}")
        if persona.goals:
            parts.append(f"你的目标：{', '.join(persona.goals)}")
        if persona.speaking_habits:
            parts.append(f"你的说话习惯：{', '.join(persona.speaking_habits)}")

        parts.append(f"你的耐心等级：{persona.patience_level}/5")

        # Legacy behavior
        parts.append(f"\n你的性格特点：")
        parts.append(f"- 态度：{persona.attitude}")
        parts.append(f"- 行为模式：{', '.join(persona.behavior_patterns)}")
        parts.append(f"- 说话风格：{persona.dialogue_style}")

        # State machine context
        if state_machine:
            parts.append(f"\n【当前状态】")
            parts.append(state_machine.get_prompt_fragment())

        # Memory context
        if memory:
            mem_ctx = memory.get_memory_context()
            if mem_ctx:
                parts.append(f"\n【对话记忆】")
                parts.append(mem_ctx)
                parts.append("\n【重要约束】")
                parts.append("- 禁止重复最近3轮已经表达过的观点。")
                parts.append("- 如果站长已经回答了你的问题，不要再次提问。")
                parts.append("- 基于你的顾虑和动机回复，不要脱离角色。")

        # General requirements
        parts.append("\n【通用要求】")
        parts.append("1. 回复必须口语化，像真实电话里说话一样自然")
        parts.append("2. 可以使用语气词（嗯、啊、吧、呢、哦）")
        parts.append("3. 可以有停顿、犹豫、重复")
        parts.append("4. 不要像机器人一样完美回答，要有真实人类的情绪")
        parts.append("5. 回复要简短，像电话里一样，15-40个字")
        parts.append("6. 不要说完整的句子，可以断断续续")
        parts.append("7. 加入一些口头禅或方言感")

        return "\n".join(parts)

    def _build_diversity_prompt(
        self,
        diversity: Optional["ResponseDiversityManager"] = None,
    ) -> str:
        """Build diversity constraint prompt."""
        if diversity:
            return diversity.get_diversity_prompt_constraint()
        return ""

    def _call_llm(self, llm_client, system_prompt: str, user_prompt: str) -> str:
        """Call LLM with the prompt."""
        if not llm_client:
            return "嗯，好的。"

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = llm_client.chat(messages, temperature=0.6, max_tokens=100)
            return response.strip()
        except Exception:
            return "嗯，知道了。"

    def _build_user_prompt(
        self,
        agent_message: str,
        context: dict,
        tone: str,
        memory: Optional["ConversationMemory"] = None,
        diversity: Optional["ResponseDiversityManager"] = None,
    ) -> str:
        """Build user prompt with history, tone, and constraints."""
        history = context.get("dialogue_history", [])
        history_text = "\n".join(
            f"{'站长' if t['role'] == 'agent' else '你'}: {t['content']}"
            for t in history[-8:]
        )

        parts = [f"对话历史：\n{history_text}\n"]
        parts.append(f"站长刚才说：\"{agent_message}\"")
        parts.append(f"\n当前态度：{tone}")

        # Diversity constraints
        div_prompt = self._build_diversity_prompt(diversity)
        if div_prompt:
            parts.append(f"\n{div_prompt}")

        parts.append("\n回复要求：")
        parts.append("- 像真实人类一样自然，有情绪和停顿")
        parts.append("- 简短口语化，15-40个字")
        parts.append("- 不要重复最近说过的话")

        return "\n".join(parts)


class CooperativeStrategy(DialogueStrategy):
    """Strategy for cooperative users - friendly but realistic."""

    def generate_response(self, agent_message, persona, context, memory=None, state_machine=None, diversity=None):
        system = self._build_system_prompt(persona, memory, state_machine)
        tone = "比较配合，但也不是完全顺从，偶尔有小担心"
        intent = "agree"

        user_prompt = self._build_user_prompt(agent_message, context, tone, memory, diversity)
        user_prompt += "\n- 可以带一点犹豫或疑问\n- 偶尔表达一点小担心或小要求\n- 不要用太正式的语气"

        # Inject style pool suggestion
        if diversity:
            suggested = diversity.pick_expression(intent)
            if suggested:
                user_prompt += f"\n- 建议你表达类似：\"{suggested}\" 的意思"

        response = self._call_llm(context.get("llm_client"), system, user_prompt)

        # Regenerate if too repetitive
        if diversity and diversity.should_regenerate(response):
            for _ in range(2):
                response = self._call_llm(context.get("llm_client"), system, user_prompt)
                if not diversity.should_regenerate(response):
                    break

        if diversity:
            diversity.track_usage(response, intent)
        return response

    def should_end_conversation(self, context: dict) -> bool:
        return context.get("commitment_obtained", False)


class RejectionStrategy(DialogueStrategy):
    """Strategy for users who initially reject - state-driven softening."""

    def generate_response(self, agent_message, persona, context, memory=None, state_machine=None, diversity=None):
        system = self._build_system_prompt(persona, memory, state_machine)

        # State-driven tone
        if state_machine:
            state = state_machine.current_state
            if state.value == "hard_reject":
                tone, intent = "坚决拒绝，带点不耐烦，可以反问", "reject"
            elif state.value == "soft_reject":
                tone, intent = "开始犹豫，语气没那么硬了，但还在观望", "hesitate"
            elif state.value == "considering":
                tone, intent = "在认真考虑，偶尔问具体问题", "question"
            elif state.value == "interested":
                tone, intent = "基本接受了，但想再确认一下", "agree"
            else:
                tone, intent = "已经同意了", "agree"
        else:
            tone, intent = "拒绝但可能软化", "reject"

        user_prompt = self._build_user_prompt(agent_message, context, tone, memory, diversity)
        user_prompt += "\n- 可以叹气、停顿、犹豫\n- 不要说得太完美，要有真实感\n- 抱怨要具体（罚款、扣钱、收入等）"

        if diversity:
            suggested = diversity.pick_expression(intent)
            if suggested:
                user_prompt += f"\n- 建议你表达类似：\"{suggested}\" 的意思"

        response = self._call_llm(context.get("llm_client"), system, user_prompt)

        if diversity and diversity.should_regenerate(response):
            for _ in range(2):
                response = self._call_llm(context.get("llm_client"), system, user_prompt)
                if not diversity.should_regenerate(response):
                    break

        if diversity:
            diversity.track_usage(response, intent)
        return response

    def should_end_conversation(self, context: dict) -> bool:
        if context.get("commitment_obtained", False):
            return True
        # Check if state machine reached committed
        sm = context.get("state_machine")
        if sm and sm.current_state.value == "committed":
            return len(context.get("dialogue_history", [])) >= 4
        return False


class EmotionalStrategy(DialogueStrategy):
    """Strategy for emotional users - need de-escalation."""

    def generate_response(self, agent_message, persona, context, memory=None, state_machine=None, diversity=None):
        system = self._build_system_prompt(persona, memory, state_machine)

        emotion_level = memory.emotion_level if memory else 3

        emotion_desc = {
            5: "非常愤怒，说话大声，可能打断对方",
            4: "很生气，语气冲，抱怨",
            3: "不满，阴阳怪气，带刺",
            2: "还有点不高兴，但平静点了",
            1: "基本平静了，可以正常对话",
        }.get(emotion_level, "平静")

        tone = f"情绪等级{emotion_level}/5（{emotion_desc}）"
        intent = "complain" if emotion_level >= 4 else "acknowledge"

        user_prompt = self._build_user_prompt(agent_message, context, tone, memory, diversity)
        user_prompt += "\n- 情绪要真实，不要突然变好\n- 可以叹气、停顿、语气词\n- 抱怨要具体（罚款、差评、扣钱等）\n- 简短，有电话对话感"

        if diversity:
            suggested = diversity.pick_expression(intent)
            if suggested:
                user_prompt += f"\n- 建议你表达类似：\"{suggested}\" 的意思"

        response = self._call_llm(context.get("llm_client"), system, user_prompt)

        if diversity and diversity.should_regenerate(response):
            for _ in range(2):
                response = self._call_llm(context.get("llm_client"), system, user_prompt)
                if not diversity.should_regenerate(response):
                    break

        if diversity:
            diversity.track_usage(response, intent)
        return response

    def should_end_conversation(self, context: dict) -> bool:
        if context.get("commitment_obtained", False):
            return True
        memory = context.get("memory")
        if memory and memory.emotion_level <= 1:
            return len(context.get("dialogue_history", [])) >= 6
        return False


class OffTopicStrategy(DialogueStrategy):
    """Strategy for users who go off topic."""

    def generate_response(self, agent_message, persona, context, memory=None, state_machine=None, diversity=None):
        system = self._build_system_prompt(persona, memory, state_machine)

        if state_machine and state_machine.current_state.value in ["considering", "interested", "committed"]:
            tone, intent = "被拉回来了，但还有点走神", "acknowledge"
        else:
            tone, intent = "继续跑题，岔开话题，突然想起别的事", "distracted"

        user_prompt = self._build_user_prompt(agent_message, context, tone, memory, diversity)
        user_prompt += "\n- 跑题要自然，像突然想起别的事\n- 可以问天气、问工资、聊家常\n- 被拉回正题时要有'哦对'的感觉\n- 简短口语化"

        if diversity:
            suggested = diversity.pick_expression(intent)
            if suggested:
                user_prompt += f"\n- 建议你表达类似：\"{suggested}\" 的意思"

        response = self._call_llm(context.get("llm_client"), system, user_prompt)

        if diversity and diversity.should_regenerate(response):
            for _ in range(2):
                response = self._call_llm(context.get("llm_client"), system, user_prompt)
                if not diversity.should_regenerate(response):
                    break

        if diversity:
            diversity.track_usage(response, intent)
        return response

    def should_end_conversation(self, context: dict) -> bool:
        sm = context.get("state_machine")
        if sm and sm.current_state.value in ["interested", "committed"]:
            return len(context.get("dialogue_history", [])) >= 6
        return False


class IndecisiveStrategy(DialogueStrategy):
    """Strategy for indecisive users - need confidence building."""

    def generate_response(self, agent_message, persona, context, memory=None, state_machine=None, diversity=None):
        system = self._build_system_prompt(persona, memory, state_machine)

        if state_machine:
            state = state_machine.current_state
            if state.value == "hard_reject":
                tone, intent = "非常犹豫，问东问西，充满担心", "hesitate"
            elif state.value == "soft_reject":
                tone, intent = "还在纠结，反复确认细节", "question"
            elif state.value == "considering":
                tone, intent = "开始认真考虑，但还有具体顾虑", "hesitate"
            elif state.value == "interested":
                tone, intent = "终于有点信心了，但还想再确认", "question"
            else:
                tone, intent = "终于决定了，但还有点担心", "agree"
        else:
            tone, intent = "犹豫不决，问东问西", "hesitate"

        user_prompt = self._build_user_prompt(agent_message, context, tone, memory, diversity)
        user_prompt += "\n- 犹豫要真实，可以反复问细节\n- 担心具体的事（罚款、时间、距离）\n- 做决定时不要干脆，要拖泥带水\n- 简短口语化"

        if diversity:
            suggested = diversity.pick_expression(intent)
            if suggested:
                user_prompt += f"\n- 建议你表达类似：\"{suggested}\" 的意思"

        response = self._call_llm(context.get("llm_client"), system, user_prompt)

        if diversity and diversity.should_regenerate(response):
            for _ in range(2):
                response = self._call_llm(context.get("llm_client"), system, user_prompt)
                if not diversity.should_regenerate(response):
                    break

        if diversity:
            diversity.track_usage(response, intent)
        return response

    def should_end_conversation(self, context: dict) -> bool:
        if context.get("commitment_obtained", False):
            return True
        sm = context.get("state_machine")
        if sm and sm.current_state.value == "committed":
            return len(context.get("dialogue_history", [])) >= 4
        return False


STRATEGY_MAP = {
    "cooperative": CooperativeStrategy,
    "rejection": RejectionStrategy,
    "emotional": EmotionalStrategy,
    "off_topic": OffTopicStrategy,
    "indecisive": IndecisiveStrategy,
}
