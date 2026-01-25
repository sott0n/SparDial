"""Minimal NumPy/SciPy tracer for CSR matmul (@ only)."""

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix

from .compiler import get_compiler


class TraceError(TypeError):
    """Raised when tracing encounters unsupported operations."""


@dataclass
class SpmvOp:
    """Traced SpMV op (CSR @ 1D)."""

    A: csr_matrix
    x: np.ndarray

    def execute(self) -> np.ndarray:
        compiler = get_compiler()
        return compiler.execute_spmv(self.A, self.x)


@dataclass
class SpmmOp:
    """Traced SpMM op (CSR @ 2D)."""

    A: csr_matrix
    B: np.ndarray

    def execute(self) -> np.ndarray:
        compiler = get_compiler()
        return compiler.execute_spmm(self.A, self.B)


class CsrProxy:
    """Proxy for CSR matrix that captures @ usage."""

    def __init__(self, A: csr_matrix):
        self.A = A

    def __matmul__(self, other):
        if isinstance(other, NumpyProxy):
            return other._matmul_csr(self.A)
        raise TraceError("Only CSR @ NumPy array is supported.")


class NumpyProxy:
    """Proxy for NumPy vector that captures @ usage."""

    def __init__(self, x: np.ndarray):
        self.x = x

    def __matmul__(self, other):
        raise TraceError("Only CSR @ NumPy array is supported.")

    def __rmatmul__(self, other):
        if isinstance(other, CsrProxy):
            return self._matmul_csr(other.A)
        raise TraceError("Only CSR @ NumPy array is supported.")

    def _matmul_csr(self, A: csr_matrix):
        if self.x.ndim == 1:
            return SpmvOp(A, self.x)
        if self.x.ndim == 2 and self.x.shape[1] == 1:
            return SpmvOp(A, self.x.ravel())
        if self.x.ndim == 2:
            return SpmmOp(A, self.x)
        raise TraceError("Only CSR @ 1D or 2D NumPy array is supported.")


def trace_numpy(fn, *args, **kwargs):
    """Trace a function that should compute CSR @ NumPy array."""
    if args and kwargs:
        raise TypeError("Use positional or keyword args, not both.")
    if kwargs:
        if "A" not in kwargs or "x" not in kwargs:
            raise TypeError("matmul requires keyword args A and x.")
        A = kwargs["A"]
        x = kwargs["x"]
    else:
        if len(args) != 2:
            raise TypeError("matmul requires exactly two arguments: A, x.")
        A, x = args

    if not isinstance(A, csr_matrix):
        raise TraceError("A must be scipy.sparse.csr_matrix.")
    if not isinstance(x, np.ndarray):
        raise TraceError("x must be a NumPy ndarray.")
    if x.ndim not in (1, 2):
        raise TraceError("x must be a 1D or 2D NumPy array.")
    if A.shape[1] != x.shape[0]:
        raise TraceError("Shape mismatch for CSR @ NumPy array.")

    A_proxy = CsrProxy(A)
    x_proxy = NumpyProxy(x)

    result = fn(A_proxy, x_proxy)
    if not isinstance(result, (SpmvOp, SpmmOp)):
        raise TraceError("Only CSR @ NumPy array is supported.")
    return result
