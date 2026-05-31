"""Sample test for bootstrap CI."""

import pytest
from outbound_eval.analytics.bootstrap import BootstrapCI


def test_bootstrap_ci_init():
    """Test bootstrap CI initialization."""
    bootstrap = BootstrapCI(n_bootstrap=100)
    assert bootstrap.n_bootstrap == 100


def test_bootstrap_calculate_ci():
    """Test bootstrap CI calculation."""
    bootstrap = BootstrapCI(n_bootstrap=1000, random_seed=42)

    samples = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    result = bootstrap.calculate_ci(samples, "rate")

    assert result is not None
    assert result.observed_value >= 0
    assert 0 <= result.ci_95_lower <= result.ci_95_upper <= 1