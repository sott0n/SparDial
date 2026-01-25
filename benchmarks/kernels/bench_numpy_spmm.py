"""NumPy SpMM (Sparse Matrix-Matrix Multiplication) Benchmark.

Compares SparDial NumPy backend vs SciPy for SpMM operations.

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
    create_numpy_matrix,
)
from spardial.numpy_backend import spmm
from spardial.numpy_backend.compiler import get_compiler


class NumpySpMMBenchmark:
    """Benchmark for NumPy backend sparse matrix-matrix multiplication.

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
        self.name = "numpy_spmm"
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
        Create CSR sparse matrix and dense matrix inputs.

        Args:
            size: (M, K, N) or (M, N) tuple
            sparsity: Sparsity level (0.0 to 1.0)
            format: Sparse format (only 'csr' supported for NumPy backend)
            dtype: Data type (np.float32 or np.float64)

        Returns:
            Tuple of (csr_matrix, dense_matrix)

        Raises:
            ValueError: If size is not 2D/3D or format is not supported
        """
        if len(size) == 3:
            m, k, n = size
        elif len(size) == 2:
            m, n = size
            k = n
        else:
            raise ValueError(
                f"NumPy SpMM benchmark requires 2D or 3D size, got {len(size)}D: {size}"
            )

        # Validate format
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"NumPy SpMM benchmark only supports formats: {self.SUPPORTED_FORMATS}, got '{format}'"
            )

        matrix = create_scipy_sparse_matrix(m, k, sparsity, dtype=dtype, seed=42)
        dense = create_numpy_matrix(k, n, dtype=dtype, seed=43)
        return matrix, dense

    def run_scipy(
        self,
        A: csr_matrix,
        B: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Run SciPy SpMM and measure time.

        Args:
            A: SciPy CSR matrix
            B: NumPy dense matrix

        Returns:
            Tuple of (output array, execution time in ms)
        """
        start = time.perf_counter()
        C = A @ B
        elapsed = (time.perf_counter() - start) * 1000
        return C, elapsed

    def run_spardial(
        self,
        A: csr_matrix,
        B: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Run SparDial NumPy backend SpMM and measure time.

        Args:
            A: SciPy CSR matrix
            B: NumPy dense matrix

        Returns:
            Tuple of (output array, execution time in ms)
        """
        start = time.perf_counter()
        C = spmm(A, B)
        elapsed = (time.perf_counter() - start) * 1000
        # Make a copy because the result may be a view of internal buffer
        return C.copy(), elapsed

    def measure_compile_time(
        self,
        A: csr_matrix,
        B: np.ndarray,
    ) -> float:
        """
        Measure compilation time by clearing cache and timing first spmm call.

        Args:
            A: SciPy CSR matrix
            B: NumPy dense matrix

        Returns:
            Compilation time in milliseconds (includes first execution)
        """
        compiler = get_compiler()
        compiler._cache.clear()

        start = time.perf_counter()
        spmm(A, B)
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
        """
        if np.isnan(scipy_output).any() or np.isinf(scipy_output).any():
            return False, float("inf"), "SciPy output has NaN/Inf"

        if np.isnan(spardial_output).any() or np.isinf(spardial_output).any():
            return False, float("inf"), "SparDial output has NaN/Inf"

        if scipy_output.shape != spardial_output.shape:
            return False, float("inf"), f"shape mismatch: {scipy_output.shape} vs {spardial_output.shape}"

        max_error = float(np.max(np.abs(scipy_output - spardial_output)))

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
        """
        result = BenchmarkResult(
            name=self.name,
            size=size,
            sparsity=sparsity,
            format=format,
        )

        A, B = self.create_inputs(size, sparsity, format)

        if verbose:
            print(f"Running benchmark: {self.name}", file=sys.stderr)
            print(f"  Size: {size}, Sparsity: {sparsity:.1%}, Format: {format}", file=sys.stderr)
            print(f"  Matrix NNZ: {A.nnz}, dtype: {A.dtype}", file=sys.stderr)

        if verbose:
            print("  SciPy warmup...", file=sys.stderr)
        for _ in range(self.warmup_iterations):
            self.run_scipy(A, B)

        if verbose:
            print("  SciPy benchmark...", file=sys.stderr)
        scipy_output = None
        for _ in range(self.benchmark_iterations):
            output, elapsed = self.run_scipy(A, B)
            result.pytorch_times.append(elapsed)
            scipy_output = output

        spardial_output = None
        if verbose:
            print("  SparDial compile time measurement...", file=sys.stderr)
            compile_time = self.measure_compile_time(A, B)
            result.spardial_compile_time = compile_time

            print("  SparDial warmup...", file=sys.stderr)
            for _ in range(self.warmup_iterations):
                self.run_spardial(A, B)

            print("  SparDial benchmark...", file=sys.stderr)
            for _ in range(self.benchmark_iterations):
                output, elapsed = self.run_spardial(A, B)
                result.spardial_times.append(elapsed)
                spardial_output = output
        else:
            with redirect_stderr(io.StringIO()):
                compile_time = self.measure_compile_time(A, B)
                result.spardial_compile_time = compile_time

                for _ in range(self.warmup_iterations):
                    self.run_spardial(A, B)

                for _ in range(self.benchmark_iterations):
                    output, elapsed = self.run_spardial(A, B)
                    result.spardial_times.append(elapsed)
                    spardial_output = output

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

        result.compute_statistics()

        if verbose:
            status = "PASS" if result.correctness_passed else "FAIL"
            print(f"  Result: {status}, Speedup: {result.speedup:.2f}x", file=sys.stderr)

        return result
