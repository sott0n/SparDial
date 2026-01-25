"""Output formatting for benchmark results."""

import json
import numpy as np
from typing import Dict, List
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

    sections = format_markdown_sections(results)
    for name, table_lines in sections.items():
        lines.append(f"### {benchmark_title(name)}")
        lines.append("")
        lines.extend(table_lines)
        lines.append("")

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
BENCHMARK_TITLES = {
    "spmv": "SpMV (Sparse Matrix-Vector Multiplication)",
    "spmm": "SpMM (Sparse Matrix-Matrix Multiplication)",
    "add": "Add (Sparse Addition)",
    # Note: For numpy_spmv, "PyTorch" column shows SciPy baseline times
    "numpy_spmv": "NumPy SpMV (SciPy baseline vs SparDial NumPy Backend)",
}


def benchmark_title(name: str) -> str:
    return BENCHMARK_TITLES.get(name, name.upper())


def _format_markdown_table(results: List[BenchmarkResult]) -> List[str]:
    if not results:
        return [
            "| Size | Sparsity | PyTorch (ms) | SparDial (ms) | Speedup | Correctness |",
            "|------|----------|--------------|---------------|---------|-------------|",
        ]

    formats = {r.format for r in results}
    include_format = len(formats) > 1

    lines = [
        "| Size | Sparsity | PyTorch (ms) | SparDial (ms) | Speedup | Correctness |",
        "|------|----------|--------------|---------------|---------|-------------|",
    ]

    def _row_key(r: BenchmarkResult):
        return (r.format, r.size, r.sparsity) if include_format else (r.size, r.sparsity)

    for r in sorted(results, key=_row_key):
        size_str = "x".join(map(str, r.size))
        if include_format:
            size_str = f"{size_str} ({r.format})"
        lines.append(
            f"| {size_str} | {r.sparsity:.0%} | {r.pytorch_mean:.3f} | "
            f"{r.spardial_mean:.3f} | {r.speedup:.2f}x | "
            f"{_correctness_cell(r)} |"
        )
    return lines


def _correctness_cell(result: BenchmarkResult) -> str:
    if result.correctness_passed:
        return "OK"
    if result.correctness_reason:
        return f"NG ({result.correctness_reason})"
    return f"NG (max_error {result.max_error:.1e})"


def format_markdown_sections(results: List[BenchmarkResult]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    order: List[str] = []
    for r in results:
        if r.name not in sections:
            sections[r.name] = []
            order.append(r.name)
        sections[r.name].append(r)

    ordered_sections: Dict[str, List[str]] = {}
    for name in order:
        ordered_sections[name] = _format_markdown_table(sections[name])
    return ordered_sections
