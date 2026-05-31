"""User persona definitions."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PersonaType(str, Enum):
    """User persona types."""

    COOPERATIVE = "cooperative"
    INDECISIVE = "indecisive"
    REJECTION = "rejection"
    EMOTIONAL = "emotional"
    OFF_TOPIC = "off_topic"
    HOSTILE = "hostile"
    MULTI_TASK = "multi_task"


class UserPersona(BaseModel):
    """Definition of a user persona for simulation."""

    persona_id: str = Field(description="Unique persona ID")
    name: str = Field(description="Persona name")
    persona_type: PersonaType = Field(description="Type of persona")

    # Behavior (legacy, kept for compatibility)
    attitude: str = Field(
        default="neutral", description="Overall attitude"
    )
    behavior_patterns: list[str] = Field(
        default_factory=list, description="Behavior patterns"
    )
    dialogue_style: str = Field(
        default="casual", description="Dialogue style"
    )
    knowledge_level: str = Field(
        default="average", description="Knowledge about the service"
    )

    # === Enhanced Persona Fields ===
    motivation: str = Field(
        default="", description="核心动机，如'赚钱'、'求稳定'"
    )
    current_status: str = Field(
        default="", description="当前状态，如'最近单量下降'"
    )
    concerns: list[str] = Field(
        default_factory=list, description="顾虑列表"
    )
    goals: list[str] = Field(
        default_factory=list, description="当前目标"
    )
    emotional_baseline: str = Field(
        default="平静", description="基础情绪状态"
    )
    patience_level: int = Field(
        default=3, ge=1, le=5, description="耐心等级 1-5"
    )
    speaking_habits: list[str] = Field(
        default_factory=list, description="说话习惯，如'喜欢反问'、'经常抱怨'"
    )

    # Response diversity pool
    response_style_pool: dict[str, list[str]] = Field(
        default_factory=dict,
        description="表达多样性池，如 {'agree': ['行', '可以试试'], 'reject': ['不想跑了']}"
    )

    # Response templates (legacy)
    response_templates: dict[str, str] = Field(
        default_factory=dict, description="Response templates"
    )

    # Difficulty weight
    difficulty_weight: float = Field(
        default=1.0, description="Difficulty multiplier (1.0=easy, 2.0=hard)"
    )

    def get_difficulty_label(self) -> str:
        """Get difficulty label based on weight."""
        if self.difficulty_weight <= 1.0:
            return "easy"
        elif self.difficulty_weight <= 1.5:
            return "medium"
        else:
            return "hard"

    def get_default_style_pool(self) -> dict[str, list[str]]:
        """Get default response style pool if none configured."""
        return {
            "agree": ["行", "可以试试", "那我看看", "应该没问题", "好吧", "听你的"],
            "reject": ["不想跑了", "算了吧", "我不考虑", "暂时没兴趣", "不用了", "别说了"],
            "hesitate": ["我再想想", "有点担心", "会不会...", "真的吗？", "那要是...", "我再问问"],
            "question": ["为啥啊？", "怎么弄？", "多少钱？", "真的假的？", "具体呢？"],
            "acknowledge": ["嗯", "知道了", "行吧", "哦", "这样啊", "好吧"],
        }


# Predefined persona templates
COOPERATIVE_PERSONA = UserPersona(
    persona_id="cooperative_base",
    name="积极配合型",
    persona_type=PersonaType.COOPERATIVE,
    attitude="positive",
    behavior_patterns=["cooperative"],
    dialogue_style="friendly",
    difficulty_weight=1.0,
    motivation="稳定收入",
    current_status="单量正常，按部就班",
    concerns=["扣款规则", "节假日补贴"],
    goals=["多跑几单", "保持好评"],
    emotional_baseline="平静",
    patience_level=4,
    speaking_habits=["直接回答", "偶尔确认细节"],
    response_style_pool={
        "agree": ["行", "没问题", "听你的", "可以可以"],
        "reject": ["这个不太行", "暂时不要吧"],
        "hesitate": ["我看看啊", "嗯...行吧"],
        "question": ["怎么操作？", "什么时候？"],
        "acknowledge": ["嗯", "知道了", "好"],
    },
)

INDECISIVE_PERSONA = UserPersona(
    persona_id="indecisive_base",
    name="犹豫不决型",
    persona_type=PersonaType.INDECISIVE,
    attitude="uncertain",
    behavior_patterns=["indecisive", "needs_encouragement"],
    dialogue_style="hesitant",
    difficulty_weight=1.3,
    motivation="想尝试但担心风险",
    current_status="观望中，没决定",
    concerns=["罚款", "时间不够", "距离太远", "怕干不好"],
    goals=["了解清楚再做决定", "想找稳妥的方案"],
    emotional_baseline="犹豫",
    patience_level=3,
    speaking_habits=["反复询问", "拖泥带水", "自我否定"],
    response_style_pool={
        "agree": ["那我试试？", "应该可以吧", "先做着看"],
        "reject": ["我再想想", "有点担心", "会不会不好啊"],
        "hesitate": ["真的吗？", "那要是...", "我再问问别人"],
        "question": ["具体怎么弄？", "多少钱？", "多久？"],
        "acknowledge": ["哦", "这样啊", "知道了"],
    },
)

REJECTION_PERSONA = UserPersona(
    persona_id="rejection_base",
    name="轻度拒绝型",
    persona_type=PersonaType.REJECTION,
    attitude="reluctant",
    behavior_patterns=["rejection", "initial_refusal"],
    dialogue_style="direct",
    difficulty_weight=1.5,
    motivation="赚钱",
    current_status="最近收入减少，心情差",
    concerns=["收入不稳定", "罚款太多", "平台扣钱厉害"],
    goals=["换平台", "减少损失"],
    emotional_baseline="不耐烦",
    patience_level=2,
    speaking_habits=["喜欢反问", "经常抱怨", "语气硬"],
    response_style_pool={
        "agree": ["行吧", "那再看看", "先这样"],
        "reject": ["不想跑了", "算了吧", "我不考虑", "暂时没兴趣", "别说了"],
        "hesitate": ["你们之前也不是这样说的", "我再想想"],
        "question": ["为啥啊？", "那之前呢？", "你们平台不也扣钱？"],
        "acknowledge": ["嗯", "知道了", "随便吧"],
    },
)

EMOTIONAL_PERSONA = UserPersona(
    persona_id="emotional_base",
    name="情绪化型",
    persona_type=PersonaType.EMOTIONAL,
    attitude="frustrated",
    behavior_patterns=["emotional", "complaint_prone"],
    dialogue_style="emotional",
    difficulty_weight=2.0,
    motivation="公平待遇",
    current_status="刚被罚款，情绪爆发",
    concerns=["罚款太多", "不被尊重", "申诉没用", "站长不管事"],
    goals=["讨个说法", "减少罚款"],
    emotional_baseline="愤怒",
    patience_level=1,
    speaking_habits=["情绪化表达", "大声说话", "打断对方", "重复抱怨"],
    response_style_pool={
        "agree": ["行吧", "那你们看着办"],
        "reject": ["我不干了！", "你们就是坑人！", "别说了！", "我不听！"],
        "hesitate": ["你们能保证吗？", "上次也是这么说的"],
        "question": ["凭啥扣我钱？", "你们管不管？", "我找谁去？"],
        "acknowledge": ["哼", "哦", "行"],
    },
)

OFF_TOPIC_PERSONA = UserPersona(
    persona_id="off_topic_base",
    name="跑题型",
    persona_type=PersonaType.OFF_TOPIC,
    attitude="distracted",
    behavior_patterns=["off_topic", "topic_switching"],
    dialogue_style="wandering",
    difficulty_weight=1.4,
    motivation="随便聊聊",
    current_status="送单中，脑子在想别的事",
    concerns=["天气不好", "路太堵", "客户难搞"],
    goals=["快点送完", "早点下班"],
    emotional_baseline="轻松",
    patience_level=4,
    speaking_habits=["跑题", "岔开话题", "突然想起别的事"],
    response_style_pool={
        "agree": ["行", "可以", "你说了算"],
        "reject": ["再说再说", "我先送单呢", "回头聊"],
        "hesitate": ["哎对了，你们那个...", "等会儿，我刚想到"],
        "question": ["今天天气咋样？", "你们站里人多吗？", "我听说..."],
        "acknowledge": ["嗯", "哦", "行吧"],
    },
)

PERSONA_TEMPLATES = {
    PersonaType.COOPERATIVE: COOPERATIVE_PERSONA,
    PersonaType.INDECISIVE: INDECISIVE_PERSONA,
    PersonaType.REJECTION: REJECTION_PERSONA,
    PersonaType.EMOTIONAL: EMOTIONAL_PERSONA,
    PersonaType.OFF_TOPIC: OFF_TOPIC_PERSONA,
}