"""NumPy SpMV (Sparse Matrix-Vector Multiplication) Benchmark.

Compares SparDial NumPy backend vs SciPy for SpMV operations.

Note: This benchmark uses BenchmarkResult.pytorch_times to store SciPy baseline
timings for compatibility with the existing reporter infrastructure. The output
labels show "PyTorch" but actually represent SciPy for this benchmark.
"""

import io
import sys
import time
from contextlib import redirect_stderr
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

from benchmarks.framework.benchmark_base import BenchmarkResult
from benchmarks.framework.numpy_sparse_utils import (
    create_scipy_sparse_matrix,
    create_numpy_vector,
)
from spardial.numpy_backend import spmv
from spardial.numpy_backend.compiler import get_compiler


class NumpySpMVBenchmark:
    """Benchmark for NumPy backend sparse matrix-vector multiplication.

    Compares SparDial NumPy backend (MLIR-compiled) against SciPy.

    Note: Only CSR format is supported by the NumPy backend.
    """

    # Supported formats for this benchmark
    SUPPORTED_FORMATS = {"csr"}

    def __init__(
        self,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ):
        """
        Initialize benchmark.

        Args:
            warmup_iterations: Number of warmup runs before timing
            benchmark_iterations: Number of timed runs
        """
        self.name = "numpy_spmv"
        self.warmup_iterations = max(0, warmup_iterations)
        self.benchmark_iterations = max(1, benchmark_iterations)

    def create_inputs(
        self,
        size: Tuple[int, ...],
        sparsity: float,
        format: str,
        dtype: np.dtype = np.float32,
    ) -> Tuple[csr_matrix, np.ndarray]:
        """
        Create CSR sparse matrix and dense vector inputs.

        Args:
            size: (rows, cols) tuple for the sparse matrix (must be 2D)
            sparsity: Sparsity level (0.0 to 1.0)
            format: Sparse format (only 'csr' supported for NumPy backend)
            dtype: Data type (np.float32 or np.float64)

        Returns:
            Tuple of (csr_matrix, dense_vector)

        Raises:
            ValueError: If size is not 2D or format is not supported
        """
        # Validate size is 2D
        if len(size) != 2:
            raise ValueError(
                f"NumPy SpMV benchmark requires 2D size (rows, cols), got {len(size)}D: {size}"
            )

        # Validate format
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"NumPy SpMV benchmark only supports formats: {self.SUPPORTED_FORMATS}, got '{format}'"
            )

        rows, cols = size
        # Use different seeds for matrix and vector to ensure they are independent
        matrix = create_scipy_sparse_matrix(rows, cols, sparsity, dtype=dtype, seed=42)
        vector = create_numpy_vector(cols, dtype=dtype, seed=43)
        return matrix, vector

    def run_scipy(
        self,
        A: csr_matrix,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Run SciPy SpMV and measure time.

        Args:
            A: SciPy CSR matrix
            x: NumPy dense vector

        Returns:
            Tuple of (output array, execution time in ms)
        """
        start = time.perf_counter()
        y = A @ x
        elapsed = (time.perf_counter() - start) * 1000
        return y, elapsed

    def run_spardial(
        self,
        A: csr_matrix,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Run SparDial NumPy backend SpMV and measure time.

        Args:
            A: SciPy CSR matrix
            x: NumPy dense vector

        Returns:
            Tuple of (output array, execution time in ms)
        """
        start = time.perf_counter()
        y = spmv(A, x)
        elapsed = (time.perf_counter() - start) * 1000
        # Make a copy because the result may be a view of internal buffer
        # that can be overwritten when the caller's context changes
        return y.copy(), elapsed

    def measure_compile_time(
        self,
        A: csr_matrix,
        x: np.ndarray,
    ) -> float:
        """
        Measure compilation time by clearing cache and timing first spmv call.

        This clears the global compiler cache to ensure we measure actual
        compilation time, not cached execution.

        Args:
            A: SciPy CSR matrix
            x: NumPy dense vector

        Returns:
            Compilation time in milliseconds (includes first execution)
        """
        # Clear the global compiler cache to measure fresh compilation
        compiler = get_compiler()
        compiler._cache.clear()

        # Time the first spmv call which includes compilation
        start = time.perf_counter()
        spmv(A, x)
        elapsed = (time.perf_counter() - start) * 1000

        return elapsed

    def check_results(
        self,
        scipy_output: np.ndarray,
        spardial_output: np.ndarray,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> Tuple[bool, float, str]:
        """
        Check correctness between SciPy and SparDial outputs.

        Args:
            scipy_output: Output from SciPy
            spardial_output: Output from SparDial
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Tuple of (passed, max_error, reason)
        """
        # Check for NaN/Inf
        if np.isnan(scipy_output).any() or np.isinf(scipy_output).any():
            return False, float("inf"), "SciPy output has NaN/Inf"

        if np.isnan(spardial_output).any() or np.isinf(spardial_output).any():
            return False, float("inf"), "SparDial output has NaN/Inf"

        # Check shape
        if scipy_output.shape != spardial_output.shape:
            return False, float("inf"), f"shape mismatch: {scipy_output.shape} vs {spardial_output.shape}"

        # Compute max absolute error
        max_error = float(np.max(np.abs(scipy_output - spardial_output)))

        # Check if results are close
        passed = bool(np.allclose(scipy_output, spardial_output, rtol=rtol, atol=atol))

        if passed:
            return True, max_error, ""
        return False, max_error, f"max_error {max_error:.1e}"

    def run(
        self,
        size: Tuple[int, ...],
        sparsity: float,
        format: str,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Run the complete benchmark.

        Args:
            size: Size of tensors (rows, cols) - must be 2D
            sparsity: Sparsity level (0.0 to 1.0)
            format: Sparse format (only 'csr' supported)
            verbose: Print detailed progress messages

        Returns:
            BenchmarkResult with timing and correctness info

        Raises:
            ValueError: If size is not 2D or format is not supported
        """
        result = BenchmarkResult(
            name=self.name,
            size=size,
            sparsity=sparsity,
            format=format,
        )

        # Create inputs (validates size and format)
        A, x = self.create_inputs(size, sparsity, format)

        if verbose:
            print(f"Running benchmark: {self.name}", file=sys.stderr)
            print(f"  Size: {size}, Sparsity: {sparsity:.1%}, Format: {format}", file=sys.stderr)
            print(f"  Matrix NNZ: {A.nnz}, dtype: {A.dtype}", file=sys.stderr)

        # SciPy warmup
        if verbose:
            print("  SciPy warmup...", file=sys.stderr)
        for _ in range(self.warmup_iterations):
            self.run_scipy(A, x)

        # SciPy benchmark
        # Note: Using pytorch_times for SciPy baseline (reporter shows "PyTorch")
        if verbose:
            print("  SciPy benchmark...", file=sys.stderr)
        scipy_output = None
        for _ in range(self.benchmark_iterations):
            output, elapsed = self.run_scipy(A, x)
            result.pytorch_times.append(elapsed)
            scipy_output = output

        # SparDial benchmark
        spardial_output = None
        if verbose:
            # Measure compile time first (with stderr visible for debugging)
            print("  SparDial compile time measurement...", file=sys.stderr)
            compile_time = self.measure_compile_time(A, x)
            result.spardial_compile_time = compile_time

            # SparDial warmup (uses cached compilation)
            print("  SparDial warmup...", file=sys.stderr)
            for _ in range(self.warmup_iterations):
                self.run_spardial(A, x)

            # SparDial benchmark
            print("  SparDial benchmark...", file=sys.stderr)
            for _ in range(self.benchmark_iterations):
                output, elapsed = self.run_spardial(A, x)
                result.spardial_times.append(elapsed)
                spardial_output = output
        else:
            # Suppress stderr during compilation
            with redirect_stderr(io.StringIO()):
                # Measure compile time first
                compile_time = self.measure_compile_time(A, x)
                result.spardial_compile_time = compile_time

                # SparDial warmup (uses cached compilation)
                for _ in range(self.warmup_iterations):
                    self.run_spardial(A, x)

                # SparDial benchmark
                for _ in range(self.benchmark_iterations):
                    output, elapsed = self.run_spardial(A, x)
                    result.spardial_times.append(elapsed)
                    spardial_output = output

        # Check correctness
        if verbose:
            print("  Checking correctness...", file=sys.stderr)
        result.correctness_passed, result.max_error, result.correctness_reason = self.check_results(
            scipy_output, spardial_output
        )

        if not result.correctness_passed and verbose:
            print(
                f"  Correctness FAIL: {result.correctness_reason}, max_error={result.max_error:.2e}",
                file=sys.stderr,
            )

        # Compute statistics
        result.compute_statistics()

        if verbose:
            status = "PASS" if result.correctness_passed else "FAIL"
            print(f"  Result: {status}, Speedup: {result.speedup:.2f}x", file=sys.stderr)

        return result
