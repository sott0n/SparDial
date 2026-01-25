"""Minimal NumPy/SciPy tracer for SpMV (@ only, CSR-only)."""

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


class CsrProxy:
    """Proxy for CSR matrix that captures @ usage."""

    def __init__(self, A: csr_matrix):
        self.A = A

    def __matmul__(self, other):
        if isinstance(other, NumpyProxy):
            return SpmvOp(self.A, other.x)
        raise TraceError("Only CSR @ 1D NumPy vector is supported.")


class NumpyProxy:
    """Proxy for NumPy vector that captures @ usage."""

    def __init__(self, x: np.ndarray):
        self.x = x

    def __matmul__(self, other):
        raise TraceError("Only CSR @ 1D NumPy vector is supported.")

    def __rmatmul__(self, other):
        if isinstance(other, CsrProxy):
            return SpmvOp(other.A, self.x)
        raise TraceError("Only CSR @ 1D NumPy vector is supported.")


def trace_spmv(fn, *args, **kwargs) -> SpmvOp:
    """Trace a function that should compute CSR @ 1D vector."""
    if args and kwargs:
        raise TypeError("Use positional or keyword args, not both.")
    if kwargs:
        if "A" not in kwargs or "x" not in kwargs:
            raise TypeError("spmv requires keyword args A and x.")
        A = kwargs["A"]
        x = kwargs["x"]
    else:
        if len(args) != 2:
            raise TypeError("spmv requires exactly two arguments: A, x.")
        A, x = args

    if not isinstance(A, csr_matrix):
        raise TraceError("A must be scipy.sparse.csr_matrix.")
    if not isinstance(x, np.ndarray):
        raise TraceError("x must be a NumPy ndarray.")
    if x.ndim == 2 and x.shape[1] == 1:
        x = x.ravel()
    if x.ndim != 1:
        raise TraceError("x must be a 1D NumPy vector.")
    if A.shape[1] != x.shape[0]:
        raise TraceError("Shape mismatch for CSR @ 1D vector.")

    A_proxy = CsrProxy(A)
    x_proxy = NumpyProxy(x)

    result = fn(A_proxy, x_proxy)
    if not isinstance(result, SpmvOp):
        raise TraceError("Only CSR @ 1D NumPy vector is supported.")
    return result
