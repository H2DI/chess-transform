#!/usr/bin/env python3
"""
Generate a Markdown report with plots and tables from an evaluation JSON.

Example:
    python tools/generate_report_markdown.py \
        --input reports/shard_00000_eval.json \
        --output reports/shard_00000_report.md
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_group_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    """Extract bucket ranges and parities from group labels."""
    buckets = []
    parities = []
    for label in labels:
        # Format: "[0,20) even" or "[0,20) odd"
        parts = label.rsplit(" ", 1)
        buckets.append(parts[0])
        parities.append(parts[1])
    return buckets, parities


def reshape_by_parity(
    values: list[float], n_buckets: int
) -> tuple[list[float], list[float]]:
    """Reshape flat list into even/odd lists (group order: b0-even, b0-odd, b1-even, ...)."""
    even = [values[i * 2] for i in range(n_buckets)]
    odd = [values[i * 2 + 1] for i in range(n_buckets)]
    return even, odd


def get_bucket_labels(group_labels: list[str]) -> list[str]:
    """Get unique bucket labels from group labels."""
    buckets, _ = parse_group_labels(group_labels)
    # Take every other one (even indices)
    return [buckets[i * 2] for i in range(len(buckets) // 2)]


def plot_metric_by_bucket_parity(
    values: list[float],
    group_labels: list[str],
    title: str,
    ylabel: str,
    output_path: str,
    higher_is_better: bool = True,
):
    """Create a grouped bar chart for a metric by bucket and parity."""
    bucket_labels = get_bucket_labels(group_labels)
    n_buckets = len(bucket_labels)
    even, odd = reshape_by_parity(values, n_buckets)

    x = np.arange(n_buckets)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, odd, width, label="White (odd ply)", color="#4A90D9")
    bars2 = ax.bar(
        x + width / 2, even, width, label="Black (even ply)", color="#D94A4A"
    )

    ax.set_xlabel("Ply Range")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_topk_accuracy(
    topk_per_group: dict[str, list[float]],
    group_labels: list[str],
    output_path: str,
):
    """Create a line plot for top-k accuracy across ply buckets."""
    bucket_labels = get_bucket_labels(group_labels)
    n_buckets = len(bucket_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for k, values in sorted(topk_per_group.items(), key=lambda x: int(x[0])):
        even, odd = reshape_by_parity(values, n_buckets)
        x = np.arange(n_buckets)

        ax1.plot(x, [v * 100 for v in odd], marker="o", label=f"Top-{k}")
        ax2.plot(x, [v * 100 for v in even], marker="o", label=f"Top-{k}")

    for ax, title in [(ax1, "White (odd ply)"), (ax2, "Black (even ply)")]:
        ax.set_xlabel("Ply Range")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Top-k Accuracy - {title}")
        ax.set_xticks(np.arange(n_buckets))
        ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_legality_by_bucket_parity(
    legality_per_group: list[float],
    group_labels: list[str],
    output_path: str,
):
    """Create a bar chart for legality rate by bucket and parity."""
    plot_metric_by_bucket_parity(
        [v * 100 for v in legality_per_group],
        group_labels,
        "Move Legality Rate by Ply Range",
        "Legality (%)",
        output_path,
        higher_is_better=True,
    )


def generate_metrics_table(metrics: dict, group_labels: list[str]) -> str:
    """Generate a markdown table for metrics per group."""
    bucket_labels = get_bucket_labels(group_labels)
    n_buckets = len(bucket_labels)

    ppl_even, ppl_odd = reshape_by_parity(metrics["perplexity_per_group"], n_buckets)
    tok_even, tok_odd = reshape_by_parity(metrics["tokens_per_group"], n_buckets)
    top1_even, top1_odd = reshape_by_parity(metrics["topk_per_group"]["1"], n_buckets)

    lines = [
        "| Ply Range | Side | Tokens | Perplexity | Top-1 Acc |",
        "|-----------|------|--------|------------|-----------|",
    ]

    for i, bucket in enumerate(bucket_labels):
        lines.append(
            f"| {bucket} | White | {tok_odd[i]:,.0f} | {ppl_odd[i]:.2f} | {top1_odd[i] * 100:.2f}% |"
        )
        lines.append(
            f"| {bucket} | Black | {tok_even[i]:,.0f} | {ppl_even[i]:.2f} | {top1_even[i] * 100:.2f}% |"
        )

    return "\n".join(lines)


def generate_topk_table(metrics: dict) -> str:
    """Generate a markdown table for overall top-k accuracy."""
    lines = [
        "| k | Accuracy |",
        "|---|----------|",
    ]
    for k, score in sorted(
        metrics["overall"]["topk_scores"].items(), key=lambda x: int(x[0])
    ):
        lines.append(f"| {k} | {score * 100:.2f}% |")
    return "\n".join(lines)


def generate_legality_table(legality: dict) -> str:
    """Generate a markdown table for legality per group."""
    group_labels = legality["group_labels"]
    bucket_labels = get_bucket_labels(group_labels)
    n_buckets = len(bucket_labels)

    leg_even, leg_odd = reshape_by_parity(legality["legality_per_group"], n_buckets)
    mov_even, mov_odd = reshape_by_parity(legality["moves_per_group"], n_buckets)

    lines = [
        "| Ply Range | Side | Moves | Legality |",
        "|-----------|------|-------|----------|",
    ]

    for i, bucket in enumerate(bucket_labels):
        lines.append(
            f"| {bucket} | White | {mov_odd[i]:,.0f} | {leg_odd[i] * 100:.2f}% |"
        )
        lines.append(
            f"| {bucket} | Black | {mov_even[i]:,.0f} | {leg_even[i] * 100:.2f}% |"
        )

    return "\n".join(lines)


def generate_markdown_report(report: dict, output_dir: Path, report_name: str) -> str:
    """Generate full markdown report content."""
    metadata = report["metadata"]
    metrics = report["metrics"]
    legality = report["legality"]
    overall = metrics["overall"]
    group_labels = metrics["group_labels"]

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    ppl_plot = f"plots/{report_name}_perplexity.png"
    topk_plot = f"plots/{report_name}_topk.png"
    legality_plot = f"plots/{report_name}_legality.png"

    plot_metric_by_bucket_parity(
        metrics["perplexity_per_group"],
        group_labels,
        "Perplexity by Ply Range",
        "Perplexity",
        output_dir / ppl_plot,
        higher_is_better=False,
    )

    plot_topk_accuracy(
        metrics["topk_per_group"],
        group_labels,
        output_dir / topk_plot,
    )

    plot_legality_by_bucket_parity(
        legality["legality_per_group"],
        group_labels,
        output_dir / legality_plot,
    )

    # Build markdown content
    md = f"""# Evaluation Report: {metadata["model_name"]}

**Generated:** {metadata["timestamp"]}

## Metadata

| Property | Value |
|----------|-------|
| Model | `{metadata["model_name"]}` |
| Data | `{metadata["data"]}` |
| Device | {metadata["device"]} |
| Batch Size | {metadata["batch_size"]} |
| Max Games | {metadata["max_games"] or "All"} |

---

## Overall Summary

| Metric | Value |
|--------|-------|
| Total Tokens | {overall["tokens"]:,.0f} |
| Perplexity | {overall["perplexity"]:.2f} |
| Legality Rate | {legality["legality_rate"] * 100:.2f}% |

### Top-k Accuracy

{generate_topk_table(metrics)}

---

## Metrics by Ply Range

### Perplexity

![Perplexity by Ply Range]({ppl_plot})

### Top-k Accuracy

![Top-k Accuracy]({topk_plot})

### Detailed Metrics Table

{generate_metrics_table(metrics, group_labels)}

---

## Legality Analysis

The legality rate measures the percentage of greedy (argmax) predictions that are legal chess moves.

![Legality by Ply Range]({legality_plot})

### Overall Legality

| Metric | Value |
|--------|-------|
| Total Moves | {legality["total_moves"]:,} |
| Legal Moves | {legality["legal_moves"]:,} |
| Illegal Moves | {legality["illegal_moves"]:,} |
| Legality Rate | {legality["legality_rate"] * 100:.2f}% |

### Legality by Ply Range

{generate_legality_table(legality)}

---

*Report generated by `tools/generate_report_markdown.py`*
"""
    return md


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown report from evaluation JSON"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to evaluation JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path (default: same as input with .md extension)",
    )
    args = parser.parse_args()

    # Load report
    with open(args.input) as f:
        report = json.load(f)

    # Determine output path
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".md")

    output_dir = output_path.parent
    report_name = input_path.stem

    # Generate markdown
    md_content = generate_markdown_report(report, output_dir, report_name)

    # Write output
    with open(output_path, "w") as f:
        f.write(md_content)

    print(f"Report generated: {output_path}")
    print(f"Plots saved to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
