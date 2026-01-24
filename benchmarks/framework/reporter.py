"""Output formatting for benchmark results."""

import json
import numpy as np
from typing import List
from benchmarks.framework.benchmark_base import BenchmarkResult


def _to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def format_human_readable(results: List[BenchmarkResult]) -> str:
    """
    Format benchmark results as human-readable table output.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        Formatted string for terminal display
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SparDial Benchmark Results")
    lines.append("=" * 80)
    lines.append("")

    if not results:
        lines.append("(no results)")
        return "\n".join(lines)

    headers = [
        "benchmark",
        "size",
        "sparsity",
        "format",
        "pytorch_mean_ms",
        "pytorch_std_ms",
        "spardial_compile_ms",
        "spardial_mean_ms",
        "spardial_std_ms",
        "speedup",
        "correct",
        "max_error",
    ]

    rows = []
    for r in results:
        rows.append([
            r.name,
            "x".join(map(str, r.size)),
            f"{r.sparsity:.1%}",
            r.format,
            f"{r.pytorch_mean:.3f}",
            f"{r.pytorch_std:.3f}",
            f"{r.spardial_compile_time:.3f}",
            f"{r.spardial_mean:.3f}",
            f"{r.spardial_std:.3f}",
            f"{r.speedup:.2f}x",
            "yes" if r.correctness_passed else "no",
            f"{r.max_error:.2e}",
        ])

    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def _format_row(values):
        return " | ".join(
            value.ljust(widths[i]) for i, value in enumerate(values)
        )

    lines.append(_format_row(headers))
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(_format_row(row))

    return "\n".join(lines)


def format_json(results: List[BenchmarkResult]) -> str:
    """
    Format benchmark results as JSON.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        JSON string
    """
    data = {
        "benchmarks": [
            _to_serializable({
                "name": r.name,
                "size": list(r.size),
                "sparsity": r.sparsity,
                "format": r.format,
                "pytorch": {
                    "mean_ms": r.pytorch_mean,
                    "std_ms": r.pytorch_std,
                    "times_ms": r.pytorch_times,
                },
                "spardial": {
                    "compile_ms": r.spardial_compile_time,
                    "mean_ms": r.spardial_mean,
                    "std_ms": r.spardial_std,
                    "times_ms": r.spardial_times,
                },
                "speedup": r.speedup,
                "correctness": {
                    "passed": r.correctness_passed,
                    "max_error": r.max_error,
                },
            })
            for r in results
        ]
    }
    return json.dumps(data, indent=2)


def format_csv(results: List[BenchmarkResult]) -> str:
    """
    Format benchmark results as CSV.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        CSV string
    """
    lines = []
    # Header
    lines.append(
        "name,size,sparsity,format,"
        "pytorch_mean_ms,pytorch_std_ms,"
        "spardial_compile_ms,spardial_mean_ms,spardial_std_ms,"
        "speedup,correctness_passed,max_error"
    )

    for r in results:
        size_str = "x".join(map(str, r.size))
        lines.append(
            f"{r.name},{size_str},{r.sparsity:.4f},{r.format},"
            f"{r.pytorch_mean:.6f},{r.pytorch_std:.6f},"
            f"{r.spardial_compile_time:.6f},{r.spardial_mean:.6f},{r.spardial_std:.6f},"
            f"{r.speedup:.4f},{r.correctness_passed},{r.max_error:.2e}"
        )

    return "\n".join(lines)
