"""Integration tests for core evaluation pipeline."""

import pytest
from outbound_eval.dataset.task import EvaluationTask, SuccessCondition, DifficultyLevel
from outbound_eval.scenarios.personas import UserPersona, PersonaType, COOPERATIVE_PERSONA
from outbound_eval.judge.engine import JudgeEngine
from outbound_eval.scenarios.memory import ConversationMemory
from outbound_eval.scenarios.state_machine import UserStateMachine, UserState


class TestJudgeEngine:
    """Integration tests for JudgeEngine."""

    def test_full_judge_pipeline(self):
        engine = JudgeEngine()
        task = EvaluationTask(
            task_id="test_integration",
            name="集成测试",
            description="测试 Judge 引擎全流程",
            skill_name="test",
            success_criteria=[
                SuccessCondition(
                    condition_id="test_cond",
                    name="测试条件",
                    description="检查 agent 是否说了确认",
                    check_type="rule",
                    check_config={"required_keywords": ["确认"]},
                    weight=1.0,
                )
            ],
            pass_threshold=0.6,
        )
        dialogue = [
            {"role": "agent", "content": "您好，这里是站长通知，您的合同已生效，请确认。"},
            {"role": "user", "content": "好的，我确认了。"},
        ]
        result = engine.evaluate(task, dialogue, agent_state={})
        assert result.task_id == "test_integration"
        assert 0 <= result.overall_score <= 100
        assert isinstance(result.passed, bool)
        assert len(result.failure_reasons) >= 0

    def test_pass_threshold_works(self):
        engine = JudgeEngine()
        task = EvaluationTask(
            task_id="test_threshold",
            name="阈值测试",
            description="测试 pass_threshold 生效",
            skill_name="test",
            success_criteria=[
                SuccessCondition(
                    condition_id="cond_1",
                    name="条件1",
                    description="测试 pass_threshold",
                    check_type="rule",
                    check_config={"required_keywords": ["不可能出现的词_xyz"]},
                    weight=1.0,
                )
            ],
            pass_threshold=0.9,
        )
        dialogue = [{"role": "agent", "content": "通知内容"}, {"role": "user", "content": "收到"}]
        result = engine.evaluate(task, dialogue, agent_state={})
        assert result.passed is False  # keyword won't match, below 0.9 threshold


class TestConversationMemory:
    """Tests for ConversationMemory."""

    def test_emotion_tracking(self):
        mem = ConversationMemory()
        assert mem.emotion_level == 3
        mem.update("太坑了，老是扣钱", "我们会改进的")
        assert mem.emotion_level == 4  # negative keywords increase
        mem.update("好的，谢谢", "感谢配合")
        assert mem.emotion_level == 3  # positive decreases

    def test_commitment_score(self):
        mem = ConversationMemory()
        assert mem.commitment_score == 0.0
        mem.update("可以，我试试", "好的")
        assert mem.commitment_score > 0  # commitment keyword detected
        mem.update("没问题，我配合", "")
        assert mem.commitment_score >= 0.3  # cumulative


class TestUserStateMachine:
    """Tests for UserStateMachine."""

    def test_initial_states(self):
        sm = UserStateMachine.from_persona_type("cooperative", patience=4)
        assert sm.current_state == UserState.INTERESTED
        sm = UserStateMachine.from_persona_type("rejection", patience=2)
        assert sm.current_state == UserState.HARD_REJECT

    def test_hard_reject_softening(self):
        sm = UserStateMachine.from_persona_type("rejection", patience=3)
        assert sm.current_state == UserState.HARD_REJECT
        # Agent says positive things
        sm.transition("我们会帮您解决问题，还有补贴可以申请")
        sm.transition("合同签约后收入会提升，我们也有保障方案")
        assert sm.current_state in [UserState.SOFT_REJECT, UserState.CONSIDERING]

    def test_patience_affects_end(self):
        # Low patience: ends quickly
        sm_low = UserStateMachine.from_persona_type("cooperative", patience=1)
        sm_low.current_state = UserState.COMMITTED
        assert sm_low.should_end_conversation([{"role": "a", "content": "x"}])  # 1 turn with committed

        # High patience: can go longer
        sm_high = UserStateMachine.from_persona_type("cooperative", patience=5)
        sm_high.current_state = UserState.INTERESTED  # not committed yet
        history = [{"role": "a", "content": "x"}] * 10
        assert not sm_high.should_end_conversation(history)  # still under max_turns=15


class TestPersonaPipeline:
    """Tests for persona-to-pipeline integration."""

    def test_persona_fields_present(self):
        p = COOPERATIVE_PERSONA
        assert p.persona_type == PersonaType.COOPERATIVE
        assert p.motivation != ""
        assert p.patience_level >= 1
        assert len(p.response_style_pool) > 0
        assert len(p.concerns) > 0

    def test_difficulty_label(self):
        easy_persona = COOPERATIVE_PERSONA.model_copy()
        easy_persona.difficulty_weight = 1.0
        assert easy_persona.get_difficulty_label() == "easy"

        hard_persona = COOPERATIVE_PERSONA.model_copy()
        hard_persona.difficulty_weight = 2.0
        assert hard_persona.get_difficulty_label() == "hard"


class TestResultSchema:
    """Tests for result data consistency."""

    def test_result_has_required_fields(self):
        """Validate that a minimal result dict has all API-required fields."""
        minimal = {
            "run_id": "test_run",
            "run_name": "test_batch",
            "task_id": "test_task",
            "scenario_id": "test_001",
            "persona_type": "cooperative",
            "difficulty": "easy",
            "overall_score": 75.0,
            "passed": True,
            "pass_threshold": 0.7,
            "task_success": 80.0,
            "flow_adherence": 75.0,
            "compliance": 80.0,
            "recovery": 70.0,
            "naturalness": 70.0,
            "state_tracking": 70.0,
            "efficiency": 70.0,
            "failure_reasons": [],
            "improvement_suggestions": [],
            "elapsed_seconds": 12.5,
            "cost": 0.01,
            "total_tokens": 1000,
        }
        # Verify API-consumed fields exist
        api_fields = ["run_id", "run_name", "task_id", "scenario_id", "persona_type",
                      "passed", "overall_score", "task_success", "flow_adherence", "compliance",
                      "failure_reasons", "improvement_suggestions", "elapsed_seconds", "cost",
                      "pass_threshold"]
        for field in api_fields:
            assert field in minimal, f"Missing required field: {field}"