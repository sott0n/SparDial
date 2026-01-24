"""Utilities for creating sparse and dense tensors for benchmarks."""

from typing import Tuple, Optional
import torch
import numpy as np


def create_sparse_matrix(
    rows: int,
    cols: int,
    sparsity: float,
    format: str = "csr",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """
    Create a sparse matrix with the specified sparsity.

    Args:
        rows: Number of rows
        cols: Number of columns
        sparsity: Sparsity level (0.0 = dense, 1.0 = all zeros)
        format: Sparse format (coo, csr, csc, dense)
        dtype: Data type
        seed: Random seed for reproducibility (None for random)

    Returns:
        PyTorch sparse tensor

    Raises:
        ValueError: If sparsity is out of range or format is unknown
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

    # Handle edge case: sparsity=1.0 means no elements, but we need at least 1
    # for sparse formats to work correctly
    if nnz == 0 and format != "dense":
        nnz = 1

    if format == "dense":
        # Create dense matrix with zeros and random values
        if nnz == 0:
            return torch.zeros(rows, cols, dtype=dtype)

        matrix = torch.zeros(rows, cols, dtype=dtype)
        indices = rng.choice(total_elements, size=nnz, replace=False)
        row_indices = indices // cols
        col_indices = indices % cols
        # Use standard normal distribution
        values = torch.from_numpy(rng.standard_normal(nnz).astype(np.float32))
        if dtype != torch.float32:
            values = values.to(dtype)
        matrix[row_indices, col_indices] = values
        return matrix

    # Generate random indices for sparse formats
    indices = rng.choice(total_elements, size=nnz, replace=False)
    row_indices = indices // cols
    col_indices = indices % cols

    # Generate random values
    values_np = rng.standard_normal(nnz).astype(np.float32)
    values = torch.from_numpy(values_np)
    if dtype != torch.float32:
        values = values.to(dtype)

    # Create COO tensor first
    indices_tensor = torch.stack([
        torch.tensor(row_indices, dtype=torch.long),
        torch.tensor(col_indices, dtype=torch.long)
    ])
    coo_tensor = torch.sparse_coo_tensor(
        indices_tensor,
        values,
        size=(rows, cols),
    ).coalesce()

    # Convert to desired format
    if format == "coo":
        return coo_tensor
    elif format == "csr":
        return coo_tensor.to_sparse_csr()
    elif format == "csc":
        return coo_tensor.to_sparse_csc()
    else:
        raise ValueError(f"Unknown sparse format: {format}. Expected: coo, csr, csc, dense")


def create_dense_vector(
    size: int,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """
    Create a dense vector with random values.

    Args:
        size: Length of the vector
        dtype: Data type
        seed: Random seed for reproducibility (None for random)

    Returns:
        PyTorch dense tensor (1D)

    Raises:
        ValueError: If size is not positive
    """
    if size <= 0:
        raise ValueError(f"Size must be positive, got {size}")

    # Use isolated random generator
    rng = np.random.default_rng(seed)
    values_np = rng.standard_normal(size).astype(np.float32)
    values = torch.from_numpy(values_np)
    if dtype != torch.float32:
        values = values.to(dtype)
    return values


def create_dense_matrix(
    rows: int,
    cols: int,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """
    Create a dense matrix with random values.

    Args:
        rows: Number of rows
        cols: Number of columns
        dtype: Data type
        seed: Random seed for reproducibility (None for random)

    Returns:
        PyTorch dense tensor (2D)

    Raises:
        ValueError: If dimensions are not positive
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Dimensions must be positive, got ({rows}, {cols})")

    # Use isolated random generator
    rng = np.random.default_rng(seed)
    values_np = rng.standard_normal((rows, cols)).astype(np.float32)
    values = torch.from_numpy(values_np)
    if dtype != torch.float32:
        values = values.to(dtype)
    return values
