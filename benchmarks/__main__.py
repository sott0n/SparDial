"""Command-line interface for running SparDial benchmarks."""

import argparse
import sys
import traceback
from typing import List, Tuple

from benchmarks.framework.benchmark_base import BenchmarkResult
from benchmarks.framework.reporter import format_human_readable, format_json, format_csv
from benchmarks.kernels.bench_spmv import SpMVBenchmark
from benchmarks.kernels.bench_spmm import SpMMBenchmark
from benchmarks.kernels.bench_add import AddBenchmark


BENCHMARKS = {
    "spmv": SpMVBenchmark,
    "spmm": SpMMBenchmark,
    "add": AddBenchmark,
}

FORMATS = ["csr", "csc", "coo", "dense"]


class ParseError(ValueError):
    """Error raised when parsing command-line arguments fails."""
    pass


def parse_sizes(sizes_str: str) -> List[Tuple[int, ...]]:
    """
    Parse size string into list of tuples.

    Args:
        sizes_str: Comma-separated size specifications (e.g., "100x100,500x500")

    Returns:
        List of size tuples

    Raises:
        ParseError: If the size string is malformed

    Examples:
        "100x100" -> [(100, 100)]
        "100x100,500x500" -> [(100, 100), (500, 500)]
        "100x200x300" -> [(100, 200, 300)]
    """
    sizes: List[Tuple[int, ...]] = []
    for size_str in sizes_str.split(","):
        size_str = size_str.strip()
        if not size_str:
            continue

        parts = size_str.split("x")
        if len(parts) < 2:
            raise ParseError(
                f"Invalid size format '{size_str}'. "
                f"Expected format: NxM (e.g., '100x100')"
            )

        try:
            dims = tuple(int(d.strip()) for d in parts)
        except ValueError as e:
            raise ParseError(
                f"Invalid size '{size_str}': dimensions must be integers. {e}"
            )

        # Validate dimensions are positive
        for i, dim in enumerate(dims):
            if dim <= 0:
                raise ParseError(
                    f"Invalid size '{size_str}': dimension {i} must be positive, got {dim}"
                )

        sizes.append(dims)

    if not sizes:
        raise ParseError("No valid sizes specified")

    return sizes


def parse_sparsities(sparsities_str: str) -> List[float]:
    """
    Parse sparsity string into list of floats.

    Args:
        sparsities_str: Comma-separated sparsity values (e.g., "0.5,0.9,0.99")

    Returns:
        List of sparsity values

    Raises:
        ParseError: If sparsity values are invalid or out of range

    Examples:
        "0.9" -> [0.9]
        "0.5,0.9,0.99" -> [0.5, 0.9, 0.99]
    """
    sparsities: List[float] = []
    for s in sparsities_str.split(","):
        s = s.strip()
        if not s:
            continue

        try:
            value = float(s)
        except ValueError as e:
            raise ParseError(f"Invalid sparsity '{s}': must be a number. {e}")

        if not (0.0 <= value <= 1.0):
            raise ParseError(
                f"Invalid sparsity {value}: must be between 0.0 and 1.0"
            )

        sparsities.append(value)

    if not sparsities:
        raise ParseError("No valid sparsities specified")

    return sparsities


def main() -> None:
    """
    Main entry point for the benchmark CLI.

    Parses command-line arguments and runs the specified benchmarks.
    Exits with code 1 if any correctness check fails.
    """
    parser = argparse.ArgumentParser(
        description="SparDial Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks --benchmarks spmv --sizes 100x100
  python -m benchmarks --benchmarks spmv,spmm --sizes 500x500 --sparsities 0.9,0.99
  python -m benchmarks --output json > results.json
  python -m benchmarks --quick
        """,
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(BENCHMARKS.keys()),
        help=f"Comma-separated list of benchmarks to run. Available: {', '.join(BENCHMARKS.keys())}",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="100x100",
        help="Comma-separated list of sizes (e.g., '100x100,500x500')",
    )
    parser.add_argument(
        "--sparsities",
        type=str,
        default="0.9",
        help="Comma-separated list of sparsity levels (e.g., '0.5,0.9,0.99')",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="csr",
        help=f"Comma-separated list of sparse formats. Available: {', '.join(FORMATS)}",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["human", "json", "csv"],
        default="human",
        help="Output format",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: smaller sizes, fewer iterations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress messages",
    )

    args = parser.parse_args()

    # Quick mode overrides (only if not explicitly set by user)
    if args.quick:
        args.sizes = "50x50"
        args.warmup = 1
        args.iterations = 3

    # Parse and validate arguments
    try:
        sizes = parse_sizes(args.sizes)
    except ParseError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        sparsities = parse_sparsities(args.sparsities)
    except ParseError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    benchmark_names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    # Validate benchmark names
    for name in benchmark_names:
        if name not in BENCHMARKS:
            print(
                f"Error: Unknown benchmark '{name}'. "
                f"Available: {', '.join(BENCHMARKS.keys())}",
                file=sys.stderr
            )
            sys.exit(1)

    # Validate formats
    for fmt in formats:
        if fmt not in FORMATS:
            print(
                f"Error: Unknown format '{fmt}'. "
                f"Available: {', '.join(FORMATS)}",
                file=sys.stderr
            )
            sys.exit(1)

    # Run benchmarks
    results: List[BenchmarkResult] = []

    for name in benchmark_names:
        benchmark_class = BENCHMARKS[name]
        benchmark = benchmark_class(
            warmup_iterations=args.warmup,
            benchmark_iterations=args.iterations,
        )

        for size in sizes:
            for sparsity in sparsities:
                for fmt in formats:
                    try:
                        result = benchmark.run(
                            size=size,
                            sparsity=sparsity,
                            format=fmt,
                            verbose=args.verbose,
                        )
                        results.append(result)
                    except Exception as e:
                        print(
                            f"Error running {name} "
                            f"(size={size}, sparsity={sparsity}, format={fmt}): {e}",
                            file=sys.stderr
                        )
                        if args.verbose:
                            traceback.print_exc()

    # Format and print results
    if args.output == "human":
        print(format_human_readable(results))
    elif args.output == "json":
        print(format_json(results))
    elif args.output == "csv":
        print(format_csv(results))

    # Return non-zero if any correctness check failed
    if results and not all(r.correctness_passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
