"""SciPy CSR to MLIR sparse runtime descriptor conversion."""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple


class SparseTensorAdapter:
    """Convert SciPy sparse matrices to MLIR sparse runtime format.

    CSR format in MLIR sparse runtime:
    - positions (crow_indices): shape (nrows+1,)
    - indices (col_indices): shape (nnz,)
    - values: shape (nnz,)
    """

    @staticmethod
    def from_csr(
        matrix: csr_matrix,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose CSR matrix into sparse components.

        Args:
            matrix: scipy.sparse.csr_matrix

        Returns:
            Tuple of (positions, indices, values) as contiguous numpy arrays
        """
        # Ensure data is canonical (sorted indices, no duplicates)
        matrix = matrix.sorted_indices()
        matrix.sum_duplicates()

        # Extract CSR components
        positions = np.ascontiguousarray(matrix.indptr)
        indices = np.ascontiguousarray(matrix.indices)
        values = np.ascontiguousarray(matrix.data)

        return positions, indices, values

    @staticmethod
    def validate_csr(matrix: csr_matrix) -> None:
        """Validate CSR matrix for MLIR sparse runtime.

        Raises:
            ValueError: if matrix is invalid for MLIR sparse runtime
        """
        if not isinstance(matrix, csr_matrix):
            raise ValueError(f"Expected csr_matrix, got {type(matrix)}")

        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {matrix.ndim}D")

        if matrix.dtype not in (np.float32, np.float64):
            raise ValueError(
                f"Unsupported dtype {matrix.dtype}. MVP supports float32/float64 only."
            )

        if matrix.indices.dtype not in (np.int32, np.int64):
            raise ValueError(
                f"Unsupported index dtype {matrix.indices.dtype}. Supported: int32, int64."
            )
