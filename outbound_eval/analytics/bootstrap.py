"""Bootstrap confidence interval calculation."""

import random
from typing import Literal, Optional
from pydantic import BaseModel, Field
from outbound_eval.infra.config import settings


class BootstrapResult(BaseModel):
    """Bootstrap CI result."""

    metric_name: str = ""
    observed_value: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    ci_99_lower: float = 0.0
    ci_99_upper: float = 0.0
    bootstrap_distribution: list[float] = Field(default_factory=list)
    bootstrap_mean: float = 0.0
    bootstrap_std: float = 0.0
    n_bootstrap: int = 10000
    n_samples: int = 0


class BootstrapComparisonResult(BaseModel):
    """Bootstrap comparison result."""

    observed_diff: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    significant: bool = False


class BootstrapCI:
    """Bootstrap confidence interval calculator."""

    def __init__(
        self,
        n_bootstrap: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        """Initialize the calculator.

        Args:
            n_bootstrap: Number of bootstrap iterations
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap or settings.bootstrap_n
        if random_seed is not None:
            random.seed(random_seed)

    def calculate_ci(
        self,
        samples: list[float],
        metric: Literal["mean", "rate", "diff"] = "mean",
        confidence_level: float = 0.95,
    ) -> BootstrapResult:
        """Calculate bootstrap confidence interval.

        Args:
            samples: List of sample values
            metric: Type of metric
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Bootstrap result with CI
        """
        n = len(samples)
        if n == 0:
            return BootstrapResult(n_samples=0)

        # Calculate observed value
        if metric == "mean":
            observed = sum(samples) / n
        elif metric == "rate":
            observed = sum(samples) / n if all(s in [0, 1] for s in samples) else sum(samples) / n
        elif metric == "diff":
            observed = samples[0] - samples[1] if len(samples) >= 2 else 0.0
        else:
            observed = sum(samples) / n

        # Bootstrap resampling
        bootstrap_values: list[float] = []
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            resample = random.choices(samples, k=n)

            # Calculate statistic
            if metric == "mean":
                boot_value = sum(resample) / n
            elif metric == "rate":
                boot_value = sum(resample) / n
            else:
                boot_value = sum(resample) / n

            bootstrap_values.append(boot_value)

        # Calculate percentiles
        alpha_95 = (1 - confidence_level) / 2
        alpha_99 = (1 - 0.99) / 2

        sorted_values = sorted(bootstrap_values)
        ci_95_lower = sorted_values[int(len(sorted_values) * alpha_95)]
        ci_95_upper = sorted_values[int(len(sorted_values) * (1 - alpha_95))]
        ci_99_lower = sorted_values[int(len(sorted_values) * alpha_99)]
        ci_99_upper = sorted_values[int(len(sorted_values) * (1 - alpha_99))]

        # Statistics
        boot_mean = sum(bootstrap_values) / len(bootstrap_values)
        boot_std = (
            sum((x - boot_mean) ** 2 for x in bootstrap_values) / len(bootstrap_values)
        ) ** 0.5

        return BootstrapResult(
            metric_name=metric,
            observed_value=observed,
            ci_95_lower=ci_95_lower,
            ci_95_upper=ci_95_upper,
            ci_99_lower=ci_99_lower,
            ci_99_upper=ci_99_upper,
            bootstrap_distribution=bootstrap_values[::10],  # Downsample for storage
            bootstrap_mean=boot_mean,
            bootstrap_std=boot_std,
            n_bootstrap=self.n_bootstrap,
            n_samples=n,
        )

    def compare_groups(
        self,
        group_a: list[float],
        group_b: list[float],
        metric: Literal["mean_diff", "rate_diff"] = "mean_diff",
    ) -> BootstrapComparisonResult:
        """Compare two groups using bootstrap.

        Args:
            group_a: First group samples
            group_b: Second group samples
            metric: Type of comparison

        Returns:
            Comparison result with CI
        """
        # Observed difference
        observed_a = sum(group_a) / len(group_a) if group_a else 0.0
        observed_b = sum(group_b) / len(group_b) if group_b else 0.0
        observed_diff = observed_a - observed_b

        # Combine for resampling
        combined = group_a + group_b

        # Bootstrap difference
        diff_values: list[float] = []
        for _ in range(self.n_bootstrap):
            resample_a = random.choices(group_a, k=len(group_a))
            resample_b = random.choices(group_b, k=len(group_b))

            diff = (sum(resample_a) / len(resample_a)) - (
                sum(resample_b) / len(resample_b)
            )
            diff_values.append(diff)

        # Percentiles
        sorted_diffs = sorted(diff_values)
        ci_95_lower = sorted_diffs[int(len(sorted_diffs) * 0.025)]
        ci_95_upper = sorted_diffs[int(len(sorted_diffs) * 0.975)]

        # Check significance (CI doesn't include 0)
        significant = ci_95_lower > 0 or ci_95_upper < 0

        return BootstrapComparisonResult(
            observed_diff=observed_diff,
            ci_95_lower=ci_95_lower,
            ci_95_upper=ci_95_upper,
            significant=significant,
        )