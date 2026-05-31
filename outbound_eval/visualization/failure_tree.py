"""Failure tree visualization."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FailureNodeType(str, Enum):
    """Types of nodes in failure tree."""

    ROOT = "root"
    CATEGORY = "category"
    METRIC = "metric"
    FAILURE_TYPE = "failure_type"
    LEAF = "leaf"


class FailureNode(BaseModel):
    """A node in the failure tree."""

    node_id: str
    node_type: FailureNodeType = FailureNodeType.ROOT
    label: str = ""
    value: float = 0.0
    percentage: float = 0.0
    parent_id: Optional[str] = None
    children: list[str] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)


class FailureTree(BaseModel):
    """Failure tree structure."""

    tree_id: str = ""
    root: FailureNode
    nodes: dict[str, FailureNode] = Field(default_factory=dict)
    total_failures: int = 0


class FailureTreeGenerator:
    """Generates failure tree for root cause analysis."""

    def generate(self, eval_results: list[dict]) -> FailureTree:
        """Generate failure tree from evaluation results.

        Args:
            eval_results: List of evaluation results

        Returns:
            Failure tree
        """
        # Count failures by type
        failure_counts: dict[str, int] = {}
        metric_counts: dict[str, dict[str, int]] = {}

        for result in eval_results:
            if result.get("passed", False):
                continue

            for reason in result.get("failure_reasons", []):
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

            # Also track by metric
            for metric in ["task_success", "flow_adherence", "compliance", "recovery"]:
                if result.get(metric, 100) < 70:
                    if metric not in metric_counts:
                        metric_counts[metric] = {}
                    metric_counts[metric][reason] = metric_counts[metric].get(reason, 0) + 1

        # Build tree
        total_failures = sum(failure_counts.values())
        nodes: dict[str, FailureNode] = {}

        # Root node
        root = FailureNode(
            node_id="root",
            node_type=FailureNodeType.ROOT,
            label=f"总失败率: {total_failures / max(len(eval_results), 1) * 100:.1f}%",
            value=total_failures,
        )
        nodes["root"] = root

        # Category nodes (by failure reason)
        for reason, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            category_id = f"cat_{reason}"
            percentage = (count / total_failures * 100) if total_failures > 0 else 0

            category_node = FailureNode(
                node_id=category_id,
                node_type=FailureNodeType.CATEGORY,
                label=f"{reason}: {count}例 ({percentage:.0f}%)",
                value=count,
                percentage=percentage,
                parent_id="root",
            )
            nodes[category_id] = category_node
            root.children.append(category_id)

        # Metric nodes
        for metric, reasons in metric_counts.items():
            metric_id = f"metric_{metric}"
            total_in_metric = sum(reasons.values())

            metric_node = FailureNode(
                node_id=metric_id,
                node_type=FailureNodeType.METRIC,
                label=f"{metric}: {total_in_metric}例",
                value=total_in_metric,
                parent_id="root",
            )
            nodes[metric_id] = metric_node
            root.children.append(metric_id)

            # Add failure reasons as children
            for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:3]:
                leaf_id = f"leaf_{metric}_{reason}"
                metric_node.children.append(leaf_id)

                leaf_node = FailureNode(
                    node_id=leaf_id,
                    node_type=FailureNodeType.LEAF,
                    label=f"{reason}: {count}例",
                    value=count,
                    parent_id=metric_id,
                )
                nodes[leaf_id] = leaf_node

        return FailureTree(
            tree_id=f"tree_{len(eval_results)}",
            root=root,
            nodes=nodes,
            total_failures=total_failures,
        )

    def to_text_format(self, tree: FailureTree, indent: int = 0) -> str:
        """Convert failure tree to text format.

        Args:
            tree: Failure tree
            indent: Current indentation level

        Returns:
            Text representation of tree
        """
        lines = []
        prefix = "  " * indent

        def add_node(node_id: str, level: int):
            node = tree.nodes.get(node_id)
            if not node:
                return

            lines.append(f"{'  ' * level}{node.label}")

            for child_id in node.children:
                add_node(child_id, level + 1)

        lines.append(tree.root.label)
        for child_id in tree.root.children:
            add_node(child_id, 1)

        return "\n".join(lines)