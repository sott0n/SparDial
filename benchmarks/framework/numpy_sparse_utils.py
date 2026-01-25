"""Utilities for creating SciPy sparse matrices and NumPy vectors for benchmarks."""

import numpy as np
from scipy.sparse import csr_matrix


def create_scipy_sparse_matrix(
    rows: int,
    cols: int,
    sparsity: float,
    dtype: np.dtype = np.float32,
    seed: int = 42,
) -> csr_matrix:
    """
    Create a SciPy CSR sparse matrix with the specified sparsity.

    Args:
        rows: Number of rows
        cols: Number of columns
        sparsity: Sparsity level (0.0 = dense, 1.0 = all zeros)
        dtype: Data type (np.float32 or np.float64)
        seed: Random seed for reproducibility

    Returns:
        SciPy CSR sparse matrix

    Raises:
        ValueError: If sparsity is out of range or dimensions are invalid
    """
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"Sparsity must be between 0.0 and 1.0, got {sparsity}")

    if rows <= 0 or cols <= 0:
        raise ValueError(f"Dimensions must be positive, got ({rows}, {cols})")

    # Use isolated random generator to avoid affecting global state
    rng = np.random.default_rng(seed)

    # Calculate number of non-zero elements
    total_elements = rows * cols
    nnz = int(total_elements * (1 - sparsity))

    # Handle edge case: sparsity=1.0 means all zeros
    if nnz == 0:
        return csr_matrix((rows, cols), dtype=dtype)

    # Generate random indices for sparse matrix
    indices = rng.choice(total_elements, size=nnz, replace=False)
    row_indices = indices // cols
    col_indices = indices % cols

    # Generate random values
    values = rng.standard_normal(nnz).astype(dtype)

    # Create CSR matrix
    return csr_matrix((values, (row_indices, col_indices)), shape=(rows, cols), dtype=dtype)


def create_numpy_vector(
    size: int,
    dtype: np.dtype = np.float32,
    seed: int = 42,
) -> np.ndarray:
    """
    Create a NumPy dense vector with random values.

    Args:
        size: Length of the vector
        dtype: Data type (np.float32 or np.float64)
        seed: Random seed for reproducibility

    Returns:
        NumPy dense array (1D)

    Raises:
        ValueError: If size is not positive
    """
    if size <= 0:
        raise ValueError(f"Size must be positive, got {size}")

    # Use isolated random generator
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size).astype(dtype)
