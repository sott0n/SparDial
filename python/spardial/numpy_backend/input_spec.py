"""Input tensor specifications for NumPy/SciPy backend."""

from dataclasses import dataclass
from typing import Tuple, Literal, Optional
import numpy as np

SparseFormat = Literal["dense", "csr"]


@dataclass
class InputSpec:
    """Input tensor specification.

    Describes the shape, dtype, and sparsity format of an input tensor.
    Used to generate appropriate MLIR types and for kernel caching.
    """

    shape: Tuple[int, ...]
    dtype: np.dtype
    format: SparseFormat = "dense"
    # CSR-specific attributes
    index_dtype: Optional[np.dtype] = None  # int32 or int64

    @classmethod
    def from_csr(cls, matrix) -> "InputSpec":
        """Create InputSpec from a SciPy CSR matrix.

        Args:
            matrix: scipy.sparse.csr_matrix

        Returns:
            InputSpec describing the CSR matrix
        """
        return cls(
            shape=matrix.shape,
            dtype=np.dtype(matrix.dtype),
            format="csr",
            index_dtype=np.dtype(matrix.indices.dtype),
        )

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "InputSpec":
        """Create InputSpec from a NumPy array.

        Args:
            array: numpy.ndarray

        Returns:
            InputSpec describing the dense array
        """
        return cls(shape=array.shape, dtype=array.dtype, format="dense")

    def signature_key(self) -> str:
        """Generate a cache key for this specification.

        Returns:
            String key suitable for caching compiled kernels
        """
        key = f"{self.shape}_{self.dtype}_{self.format}"
        if self.index_dtype is not None:
            key += f"_{self.index_dtype}"
        return key
