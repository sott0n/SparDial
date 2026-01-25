"""High-level API for NumPy/SciPy backend (CSR MVP)."""

import numpy as np
from scipy.sparse import csr_matrix

from .compiler import get_compiler


def spmv(A, x) -> np.ndarray:
    """Sparse Matrix-Vector Multiplication: y = A @ x

    Compiles the SpMV operation to optimized LLVM code via MLIR
    sparse tensor infrastructure and executes it.

    Args:
        A: scipy.sparse.csr_matrix (or other format that can be converted)
        x: NumPy dense vector (shape (n,) or (n,1))

    Returns:
        np.ndarray: Result vector (shape (m,))

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> import numpy as np
        >>> from spardial import spmv
        >>>
        >>> A_dense = np.array([
        ...     [0, 0, 1, 0],
        ...     [2, 0, 0, 0],
        ...     [0, 3, 0, 4],
        ...     [0, 0, 0, 0],
        ... ], dtype=np.float32)
        >>> A = csr_matrix(A_dense)
        >>> x = np.array([1, 2, 3, 4], dtype=np.float32)
        >>> y = spmv(A, x)
        >>> print(y)  # [ 3.  2. 22.  0.]

    Notes:
        - MVP supports CSR format only. Other formats will be converted.
        - Supported dtypes: float32, float64
        - Supported index dtypes: int32, int64
    """
    # Convert other sparse formats to CSR
    if hasattr(A, "tocsr") and not isinstance(A, csr_matrix):
        A = A.tocsr()

    compiler = get_compiler()
    return compiler.execute_spmv(A, x)


def spmm(A, B) -> np.ndarray:
    """Sparse Matrix-Matrix Multiplication: C = A @ B

    Compiles the SpMM operation to optimized LLVM code via MLIR
    sparse tensor infrastructure and executes it.

    Args:
        A: scipy.sparse.csr_matrix (or other format that can be converted)
        B: NumPy dense matrix (shape (n, k))

    Returns:
        np.ndarray: Result matrix (shape (m, k))

    Notes:
        - MVP supports CSR format only. Other formats will be converted.
        - Supported dtypes: float32, float64
        - Supported index dtypes: int32, int64
    """
    # Convert other sparse formats to CSR
    if hasattr(A, "tocsr") and not isinstance(A, csr_matrix):
        A = A.tocsr()

    compiler = get_compiler()
    return compiler.execute_spmm(A, B)
