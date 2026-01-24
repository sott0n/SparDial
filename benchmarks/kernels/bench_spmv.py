"""SpMV (Sparse Matrix-Vector Multiplication) Benchmark."""

from typing import Tuple
import torch

from benchmarks.framework.benchmark_base import BenchmarkBase
from benchmarks.framework.sparse_utils import create_sparse_matrix, create_dense_vector
from spardial.models.kernels import MVNet


class SpMVBenchmark(BenchmarkBase):
    """Benchmark for sparse matrix-vector multiplication."""

    def __init__(
        self,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ):
        super().__init__(
            name="spmv",
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
        )

    def create_model(self) -> torch.nn.Module:
        """Create MVNet model."""
        return MVNet()

    def create_inputs(
        self,
        size: Tuple[int, ...],
        sparsity: float,
        format: str,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create sparse matrix and dense vector inputs.

        Args:
            size: (rows, cols) tuple for the sparse matrix
            sparsity: Sparsity level (0.0 to 1.0)
            format: Sparse format (coo, csr, csc, dense)

        Returns:
            Tuple of (sparse_matrix, dense_vector)
        """
        rows, cols = size
        matrix = create_sparse_matrix(rows, cols, sparsity, format)
        vector = create_dense_vector(cols)
        return (matrix, vector)
