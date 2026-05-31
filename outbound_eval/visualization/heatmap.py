"""Failure heatmap visualization."""

from typing import Optional
from pydantic import BaseModel, Field


class HeatmapCell(BaseModel):
    """A single cell in the heatmap."""

    x: int = 0
    y: int = 0
    value: float = 0.0
    color: str = "#FFFFFF"
    tooltip: str = ""


class HeatmapData(BaseModel):
    """Heatmap data for visualization."""

    title: str = ""
    x_labels: list[str] = Field(default_factory=list)
    y_labels: list[str] = Field(default_factory=list)
    cells: list[HeatmapCell] = Field(default_factory=list)


class FailureHeatmap:
    """Generates failure heatmaps for analysis."""

    # Color scheme: green (low failure) to red (high failure)
    COLOR_SCHEME = [
        "#4CAF50",  # 0-20% failure (green)
        "#8BC34A",  # 20-40%
        "#CDDC39",  # 40-50%
        "#FFC107",  # 50-60% (yellow)
        "#FF9800",  # 60-80%
        "#F44336",  # 80-100% failure (red)
    ]

    def generate_by_scenario_metric(
        self,
        eval_results: list[dict],
    ) -> HeatmapData:
        """Generate scenario × metric failure heatmap.

        Args:
            eval_results: List of evaluation results

        Returns:
            Heatmap data
        """
        # Get unique scenarios and metrics
        scenarios = list(set(r.get("task_id", "unknown") for r in eval_results))
        metrics = ["task_success", "flow_adherence", "compliance", "recovery", "naturalness"]

        x_labels = sorted(scenarios)
        y_labels = metrics

        cells: list[HeatmapCell] = []

        for x_idx, scenario in enumerate(x_labels):
            for y_idx, metric in enumerate(y_labels):
                # Filter results for this scenario
                scenario_results = [r for r in eval_results if r.get("task_id") == scenario]

                # Calculate failure rate for this metric
                metric_scores = [
                    r.get(metric, 100.0) for r in scenario_results
                ]
                failure_rate = 1 - (sum(metric_scores) / len(metric_scores) / 100) if metric_scores else 0
                failure_rate *= 100

                cells.append(
                    HeatmapCell(
                        x=x_idx,
                        y=y_idx,
                        value=failure_rate,
                        color=self._get_color(failure_rate),
                        tooltip=f"{scenario} - {metric}: {failure_rate:.1f}% 失败率",
                    )
                )

        return HeatmapData(
            title="场景 × 指标 失败热力图",
            x_labels=x_labels,
            y_labels=y_labels,
            cells=cells,
        )

    def generate_by_persona_metric(
        self,
        eval_results: list[dict],
    ) -> HeatmapData:
        """Generate persona × metric failure heatmap.

        Args:
            eval_results: List of evaluation results

        Returns:
            Heatmap data
        """
        personas = list(set(r.get("persona_type", "unknown") for r in eval_results))
        metrics = ["task_success", "flow_adherence", "compliance", "recovery", "naturalness"]

        x_labels = sorted(personas)
        y_labels = metrics

        cells: list[HeatmapCell] = []

        for x_idx, persona in enumerate(x_labels):
            for y_idx, metric in enumerate(y_labels):
                persona_results = [r for r in eval_results if r.get("persona_type") == persona]

                metric_scores = [r.get(metric, 100.0) for r in persona_results]
                failure_rate = 1 - (sum(metric_scores) / len(metric_scores) / 100) if metric_scores else 0
                failure_rate *= 100

                cells.append(
                    HeatmapCell(
                        x=x_idx,
                        y=y_idx,
                        value=failure_rate,
                        color=self._get_color(failure_rate),
                        tooltip=f"{persona} - {metric}: {failure_rate:.1f}% 失败率",
                    )
                )

        return HeatmapData(
            title="用户画像 × 指标 失败热力图",
            x_labels=x_labels,
            y_labels=y_labels,
            cells=cells,
        )

    def _get_color(self, failure_rate: float) -> str:
        """Get color for failure rate.

        Args:
            failure_rate: Failure rate percentage (0-100)

        Returns:
            Hex color string
        """
        if failure_rate <= 20:
            return self.COLOR_SCHEME[0]
        elif failure_rate <= 40:
            return self.COLOR_SCHEME[1]
        elif failure_rate <= 50:
            return self.COLOR_SCHEME[2]
        elif failure_rate <= 60:
            return self.COLOR_SCHEME[3]
        elif failure_rate <= 80:
            return self.COLOR_SCHEME[4]
        else:
            return self.COLOR_SCHEME[5]