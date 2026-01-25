"""NumPy/SciPy backend for SparDial (CSR MVP).

This module provides direct MLIR-based sparse tensor operations
for NumPy arrays and SciPy sparse matrices, without requiring PyTorch.
"""

from .api import spmv
from .compiler import SparseCompiler, get_compiler
from .kernel_builder import KernelBuilder, KernelType
from .input_spec import InputSpec
from .sparse_encoding import SparseEncodingBuilder
from .sparse_adapter import SparseTensorAdapter

__all__ = [
    # High-level API
    "spmv",
    # Classes
    "SparseCompiler",
    "KernelBuilder",
    "KernelType",
    "InputSpec",
    "SparseEncodingBuilder",
    "SparseTensorAdapter",
    # Utilities
    "get_compiler",
]
