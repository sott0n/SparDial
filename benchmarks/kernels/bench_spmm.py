"""SpMM (Sparse Matrix-Matrix Multiplication) Benchmark."""

from typing import Tuple
import torch

from benchmarks.framework.benchmark_base import BenchmarkBase
from benchmarks.framework.sparse_utils import create_sparse_matrix
from spardial.models.kernels import MMNet


class SpMMBenchmark(BenchmarkBase):
    """Benchmark for sparse matrix-matrix multiplication."""

    def __init__(
        self,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ):
        super().__init__(
            name="spmm",
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
        )

    def create_model(self) -> torch.nn.Module:
        """Create MMNet model."""
        return MMNet()

    def create_inputs(
        self,
        size: Tuple[int, ...],
        sparsity: float,
        format: str,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create two sparse matrices for multiplication.

        Args:
            size: (M, K, N) tuple where A is MxK and B is KxN
                  or (M, N) tuple where both are MxN (for square matrices)
            sparsity: Sparsity level (0.0 to 1.0)
            format: Sparse format (coo, csr, csc, dense)

        Returns:
            Tuple of (matrix_a, matrix_b)
        """
        if len(size) == 3:
            m, k, n = size
        else:
            m, n = size
            k = n

        # First matrix: M x K
        matrix_a = create_sparse_matrix(m, k, sparsity, format, seed=42)
        # Second matrix: K x N
        matrix_b = create_sparse_matrix(k, n, sparsity, format, seed=43)

        return (matrix_a, matrix_b)
