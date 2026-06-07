"""Experiment comparison with statistical analysis."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from outbound_eval.experiments.runner import ExperimentRun
from outbound_eval.analytics.bootstrap import BootstrapCI


class DifficultyDelta(BaseModel):
    """Per-difficulty delta between baseline and candidate runs."""

    difficulty: str
    success_rate_delta: float = 0.0
    score_delta: float = 0.0
    sample_size: int = 0
    p_value: float = 1.0
    is_significant: bool = False
    effect_size: float = 0.0
    baseline_success_rate: float = 0.0
    candidate_success_rate: float = 0.0


class ExperimentComparison(BaseModel):
    """Comparison result between two experiment runs."""

    experiment_id: str = ""
    baseline_run: ExperimentRun
    candidate_run: ExperimentRun

    # Point estimates (overall, all levels pooled)
    success_rate_delta: float = 0.0
    score_delta: float = 0.0
    cost_delta: float = 0.0

    # Bootstrap CI
    success_rate_delta_ci_95: tuple[float, float] = (0.0, 0.0)
    score_delta_ci_95: tuple[float, float] = (0.0, 0.0)

    # Statistical test
    p_value: float = 1.0
    is_significant: bool = False
    effect_size: float = 0.0

    # Recommendation
    recommendation: Literal["adopt_candidate", "keep_baseline", "inconclusive"] = "inconclusive"
    confidence: float = 0.0
    reasoning: str = ""

    # v2: per-difficulty matrix
    difficulty_levels: list[str] = Field(default_factory=list)
    per_difficulty: dict[str, DifficultyDelta] = Field(default_factory=dict)
    is_cross_business: bool = False
    baseline_task_id: str = ""
    candidate_task_id: str = ""


class ExperimentComparisonAnalyzer:
    """Analyzes experiment comparison results."""

    def __init__(self):
        """Initialize the analyzer."""
        self.bootstrap = BootstrapCI()

    def analyze(
        self,
        baseline_run: ExperimentRun,
        candidate_run: ExperimentRun,
        experiment_id: str = "",
        is_cross_business: bool = False,
        baseline_task_id: str = "",
        candidate_task_id: str = "",
    ) -> ExperimentComparison:
        """Analyze and compare two experiment runs.

        Args:
            baseline_run: Baseline version results (may carry
                ``results_by_difficulty`` for per-level analysis)
            candidate_run: Candidate version results
            experiment_id: Experiment ID
            is_cross_business: True if baseline and candidate are different
                business tasks (only for warning UI; stats still computed)

        Returns:
            Comparison result
        """
        # ------------------------------------------------------------------
        # Per-difficulty analysis (v2)
        # ------------------------------------------------------------------
        levels = self._intersect_levels(baseline_run, candidate_run)
        per_diff: dict[str, DifficultyDelta] = {}
        for lvl in levels:
            b_res = (baseline_run.results_by_difficulty or {}).get(lvl, [])
            c_res = (candidate_run.results_by_difficulty or {}).get(lvl, [])
            dd = self._diff_for_level(lvl, b_res, c_res)
            per_diff[lvl] = dd

        # ------------------------------------------------------------------
        # Overall analysis (uses real per-result success flags when v2 is
        # populated; otherwise falls back to run-level summary numbers)
        # ------------------------------------------------------------------
        baseline_flags = self._success_flags(baseline_run)
        candidate_flags = self._success_flags(candidate_run)

        if baseline_flags and candidate_flags:
            comparison = self.bootstrap.compare_groups(
                baseline_flags, candidate_flags, "rate_diff"
            )
            success_rate_ci = (comparison.ci_95_lower, comparison.ci_95_upper)
            overall_significant = comparison.significant
        else:
            # Fallback when no per-result data (legacy run objects)
            comparison = None
            success_rate_ci = (0.0, 0.0)
            overall_significant = False

        p_value, effect_size = self._compute_p_effect(
            baseline_flags, candidate_flags
        )

        success_rate_delta = candidate_run.success_rate - baseline_run.success_rate
        score_delta = candidate_run.avg_score - baseline_run.avg_score
        cost_delta = candidate_run.avg_cost - baseline_run.avg_cost

        # Recommendation
        if overall_significant and success_rate_delta > 0:
            recommendation: Literal["adopt_candidate", "keep_baseline", "inconclusive"] = "adopt_candidate"
            confidence = 0.9 if p_value < 0.01 else 0.7
        elif overall_significant and success_rate_delta < 0:
            recommendation = "keep_baseline"
            confidence = 0.9 if p_value < 0.01 else 0.7
        else:
            recommendation = "inconclusive"
            confidence = 0.5

        # Reasoning: prefer per-difficulty story when available
        if per_diff:
            reasoning = self._reasoning_from_matrix(
                per_diff, success_rate_delta, p_value, overall_significant
            )
        else:
            reasoning = self._generate_reasoning(
                success_rate_delta, p_value, effect_size, overall_significant
            )

        return ExperimentComparison(
            experiment_id=experiment_id,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            success_rate_delta=success_rate_delta,
            score_delta=score_delta,
            cost_delta=cost_delta,
            success_rate_delta_ci_95=success_rate_ci,
            p_value=p_value,
            is_significant=overall_significant,
            effect_size=effect_size,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            difficulty_levels=levels,
            per_difficulty=per_diff,
            is_cross_business=is_cross_business,
            baseline_task_id=baseline_task_id,
            candidate_task_id=candidate_task_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _intersect_levels(b: ExperimentRun, c: ExperimentRun) -> list[str]:
        b_levels = b.difficulty_levels or list((b.results_by_difficulty or {}).keys())
        c_levels = c.difficulty_levels or list((c.results_by_difficulty or {}).keys())
        if not b_levels or not c_levels:
            return []
        return [lvl for lvl in b_levels if lvl in set(c_levels)]

    @staticmethod
    def _success_flags(run: ExperimentRun) -> list[float]:
        """Extract per-result success flags from ``results_by_difficulty``.

        Falls back to constructing flags from the run's ``success_rate`` and
        ``total_samples`` for legacy run objects without per-level data.
        """
        flags: list[float] = []
        for rs in (run.results_by_difficulty or {}).values():
            for r in rs:
                passed = r.get("passed", None)
                if passed is None:
                    continue
                flags.append(1.0 if passed else 0.0)
        if flags:
            return flags
        # Legacy fallback: synthesize from totals
        if run.total_samples > 0:
            return [1.0] * run.passed_samples + [0.0] * (run.total_samples - run.passed_samples)
        return []

    def _diff_for_level(
        self, level: str, baseline_results: list[dict], candidate_results: list[dict]
    ) -> DifficultyDelta:
        b_flags = [1.0 if r.get("passed") else 0.0 for r in baseline_results]
        c_flags = [1.0 if r.get("passed") else 0.0 for r in candidate_results]
        b_score = [r.get("overall_score", 0.0) for r in baseline_results if "overall_score" in r]
        c_score = [r.get("overall_score", 0.0) for r in candidate_results if "overall_score" in r]
        b_rate = (sum(b_flags) / len(b_flags) * 100) if b_flags else 0.0
        c_rate = (sum(c_flags) / len(c_flags) * 100) if c_flags else 0.0
        b_avg = (sum(b_score) / len(b_score)) if b_score else 0.0
        c_avg = (sum(c_score) / len(c_score)) if c_score else 0.0

        p_value = 1.0
        is_sig = False
        effect = 0.0
        if b_flags and c_flags:
            comp = self.bootstrap.compare_groups(b_flags, c_flags, "rate_diff")
            is_sig = comp.significant
            p_value, effect = self._compute_p_effect(b_flags, c_flags)

        return DifficultyDelta(
            difficulty=level,
            success_rate_delta=c_rate - b_rate,
            score_delta=c_avg - b_avg,
            sample_size=len(b_flags) + len(c_flags),
            p_value=p_value,
            is_significant=is_sig,
            effect_size=effect,
            baseline_success_rate=b_rate,
            candidate_success_rate=c_rate,
        )

    @staticmethod
    def _compute_p_effect(
        b_flags: list[float], c_flags: list[float]
    ) -> tuple[float, float]:
        import math
        n1, n2 = len(b_flags), len(c_flags)
        if not b_flags or not c_flags:
            return 1.0, 0.0
        p1 = sum(b_flags) / n1
        p2 = sum(c_flags) / n2
        pooled = (sum(b_flags) + sum(c_flags)) / (n1 + n2)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)) if pooled > 0 else 0.0
        if se > 0:
            z = abs(p1 - p2) / se
            p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        else:
            p = 1.0

        def phi(pp):
            return 2 * math.asin(math.sqrt(max(0.001, min(0.999, pp))))

        effect = abs(phi(p2) - phi(p1)) if (p1 > 0 or p2 > 0) else 0.0
        return p, effect

    @staticmethod
    def _reasoning_from_matrix(
        per_diff: dict[str, "DifficultyDelta"],
        overall_delta: float,
        p_value: float,
        overall_sig: bool,
    ) -> str:
        if not per_diff:
            return "差异不显著，无法得出确定结论。"
        sig_wins = [d for d in per_diff.values() if d.is_significant and d.success_rate_delta > 0]
        sig_loses = [d for d in per_diff.values() if d.is_significant and d.success_rate_delta < 0]
        if sig_wins and not sig_loses:
            levels = "、".join(d.difficulty for d in sig_wins)
            return f"候选版本在 {levels} 难度下显著优于基线（p<0.05），建议采用。"
        if sig_loses and not sig_wins:
            levels = "、".join(d.difficulty for d in sig_loses)
            return f"候选版本在 {levels} 难度下显著差于基线（p<0.05），建议保持基线。"
        if sig_wins and sig_loses:
            w = "、".join(d.difficulty for d in sig_wins)
            l = "、".join(d.difficulty for d in sig_loses)
            return f"候选版本在不同难度下表现分化：{w} 显著更优，{l} 显著更差，建议按难度分场景使用。"
        return f"各难度档差异均不显著（整体 Δ={overall_delta:+.2f}%, p={p_value:.3f}），结论不确定。"

    @staticmethod
    def _generate_reasoning(
        delta: float,
        p_value: float,
        effect_size: float,
        significant: bool,
    ) -> str:
        if not significant:
            return "差异不显著，无法得出确定结论。"

        if delta > 0:
            if p_value < 0.01:
                return f"候选版本显著优于基线版本（p={p_value:.3f}），建议采用。"
            else:
                return f"候选版本优于基线版本（p={p_value:.3f}），建议考虑采用。"
        else:
            if p_value < 0.01:
                return f"候选版本显著差于基线版本（p={p_value:.3f}），建议保持基线版本。"
            else:
                return f"候选版本差于基线版本（p={p_value:.3f}），建议保持基线版本。"
