"""Sample test for persona metrics."""

import pytest
from outbound_eval.analytics.persona_metrics import PersonaMetricsAnalyzer


def test_persona_metrics_analyzer_init():
    """Test persona metrics analyzer initialization."""
    analyzer = PersonaMetricsAnalyzer()
    assert analyzer is not None


def test_persona_metrics_analyze():
    """Test persona metrics analysis."""
    analyzer = PersonaMetricsAnalyzer()

    results = [
        {"task_id": "test", "persona_type": "cooperative", "passed": True, "overall_score": 85.0},
        {"task_id": "test", "persona_type": "cooperative", "passed": True, "overall_score": 90.0},
        {"task_id": "test", "persona_type": "rejection", "passed": False, "overall_score": 60.0},
    ]

    report = analyzer.analyze(results)

    assert report is not None
    assert len(report.persona_metrics) == 2
    assert report.avg_success_rate >= 0