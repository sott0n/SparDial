"""Sparse Addition Benchmark."""

from typing import Tuple
import torch

from benchmarks.framework.benchmark_base import BenchmarkBase
from benchmarks.framework.sparse_utils import create_sparse_matrix
from spardial.models.kernels import AddNet


class AddBenchmark(BenchmarkBase):
    """Benchmark for sparse tensor addition."""

    def __init__(
        self,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ):
        super().__init__(
            name="add",
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
        )

    def create_model(self) -> torch.nn.Module:
        """Create AddNet model."""
        return AddNet()

    def create_inputs(
        self,
        size: Tuple[int, ...],
        sparsity: float,
        format: str,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create two sparse matrices for addition.

        Args:
            size: (rows, cols) tuple for the matrices
            sparsity: Sparsity level (0.0 to 1.0)
            format: Sparse format (coo, csr, csc, dense)

        Returns:
            Tuple of (matrix_a, matrix_b)
        """
        rows, cols = size

        # Create two sparse matrices with different seeds for different sparsity patterns
        matrix_a = create_sparse_matrix(rows, cols, sparsity, format, seed=42)
        matrix_b = create_sparse_matrix(rows, cols, sparsity, format, seed=43)

        return (matrix_a, matrix_b)
