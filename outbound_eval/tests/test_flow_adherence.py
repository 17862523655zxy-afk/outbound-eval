"""Sample test for flow adherence judge."""

import pytest
from outbound_eval.judge.flow_adherence import FlowAdherenceJudge


def test_flow_adherence_judge_init():
    """Test flow adherence judge initialization."""
    flow_steps = [
        {"id": "step1", "description": "Step 1"},
        {"id": "step2", "description": "Step 2"},
    ]
    judge = FlowAdherenceJudge(flow_steps)

    assert judge is not None
    assert len(judge.flow_steps) == 2


def test_flow_adherence_evaluate():
    """Test flow adherence evaluation."""
    flow_steps = [
        {"id": "step1", "description": "Confirm identity", "expected_keywords": {"required": ["请问", "你是"]}},
        {"id": "step2", "description": "Deliver message", "expected_keywords": {"required": ["合同", "生效"]}},
    ]

    judge = FlowAdherenceJudge(flow_steps)

    dialogue_history = [
        {"role": "agent", "content": "你好，请问是张明吗？"},
        {"role": "user", "content": "是"},
        {"role": "agent", "content": "好的，合同今天已生效。"},
        {"role": "user", "content": "知道了。"},
    ]

    result = judge.evaluate(dialogue_history)

    assert result is not None
    assert result.completed_steps >= 0
    assert 0 <= result.adherence_score <= 100