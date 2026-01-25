"""Sparse kernel compilation and execution (CSR MVP)."""

from typing import Dict, List, Any
import sys
import numpy as np
from scipy.sparse import csr_matrix

from spardial import ir
from spardial.pipeline import sparsify_and_bufferize, prepare_for_execution
from spardial.jit_runtime import SparDialInvoker, find_executable_function

from .input_spec import InputSpec
from .kernel_builder import KernelBuilder, KernelType
from .sparse_adapter import SparseTensorAdapter


class SparseCompiler:
    """Compiler for sparse kernels."""

    def __init__(self):
        self._kernel_builder = KernelBuilder()
        self._cache: Dict[str, Any] = {}

    def _make_cache_key(
        self,
        kernel_type: KernelType,
        input_specs: List[InputSpec],
        output_spec: InputSpec,
    ) -> str:
        """Generate cache key (includes index_dtype)."""
        input_keys = "_".join(spec.signature_key() for spec in input_specs)
        output_key = output_spec.signature_key()
        return f"{kernel_type.name}_{input_keys}_{output_key}"

    def compile(
        self,
        kernel_type: KernelType,
        input_specs: List[InputSpec],
        output_spec: InputSpec,
    ) -> ir.Module:
        """Compile kernel.

        Args:
            kernel_type: Type of kernel
            input_specs: Input specifications
            output_spec: Output specification

        Returns:
            Compiled MLIR module ready for execution
        """
        cache_key = self._make_cache_key(kernel_type, input_specs, output_spec)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build MLIR module
        module = self._kernel_builder.build(kernel_type, input_specs, output_spec)

        print(f"Generated MLIR module:", file=sys.stderr)
        print(module, file=sys.stderr)

        # Apply full sparse compilation pipeline
        module = self._apply_sparse_pipeline(module)

        print(f"After sparse pipeline:", file=sys.stderr)
        print(module, file=sys.stderr)

        self._cache[cache_key] = module
        return module

    def _apply_sparse_pipeline(self, module: ir.Module) -> ir.Module:
        """Apply sparse compilation pipeline and prepare for execution."""
        module = sparsify_and_bufferize(module)
        module = prepare_for_execution(module)
        return module

    def execute_spmv(
        self,
        A: csr_matrix,
        x: np.ndarray,
    ) -> np.ndarray:
        """Execute SpMV: y = A @ x

        Args:
            A: SciPy CSR matrix
            x: NumPy dense vector

        Returns:
            np.ndarray: Result vector
        """
        # Validate inputs
        SparseTensorAdapter.validate_csr(A)

        if x.ndim == 2 and x.shape[1] == 1:
            x = x.ravel()  # (n,1) -> (n,)

        if x.ndim != 1:
            raise ValueError(f"x must be 1D vector, got shape {x.shape}")

        if A.shape[1] != x.shape[0]:
            raise ValueError(
                f"Shape mismatch: A is {A.shape}, x is {x.shape}"
            )

        # Ensure contiguous arrays with matching dtype
        if x.dtype != A.dtype:
            x = x.astype(A.dtype)
        x = np.ascontiguousarray(x)

        # Create input specs
        A_spec = InputSpec.from_csr(A)
        x_spec = InputSpec.from_numpy(x)
        y_spec = InputSpec(
            shape=(A.shape[0],),
            dtype=A.dtype,
            format="dense"
        )

        # Compile (or get from cache)
        module = self.compile(KernelType.SPMV, [A_spec, x_spec], y_spec)

        # Prepare output buffer (initialized to zero for accumulation)
        y = np.zeros(A.shape[0], dtype=A.dtype)

        # Invoke with sparse tensor components
        result = self._invoke_spmv(module, A, x, y)

        return result

    def _invoke_spmv(
        self,
        module: ir.Module,
        A: csr_matrix,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Invoke SpMV kernel using the shared ExecutionEngine path."""
        positions, indices, values = SparseTensorAdapter.from_csr(A)

        positions = np.ascontiguousarray(positions.astype(A.indices.dtype))
        indices = np.ascontiguousarray(indices.astype(A.indices.dtype))
        values = np.ascontiguousarray(values.astype(A.dtype))

        print(f"Executing function: spmv", file=sys.stderr)
        print(f"Positions: {positions} shape={positions.shape} dtype={positions.dtype}", file=sys.stderr)
        print(f"Indices: {indices} shape={indices.shape} dtype={indices.dtype}", file=sys.stderr)
        print(f"Values: {values} shape={values.shape} dtype={values.dtype}", file=sys.stderr)
        print(f"x: {x} shape={x.shape} dtype={x.dtype}", file=sys.stderr)
        print(f"y: {y} shape={y.shape} dtype={y.dtype}", file=sys.stderr)

        invoker = SparDialInvoker(module, opt_level=0)
        function_name = find_executable_function(module)

        return invoker.invoke(function_name, positions, indices, values, x, y)


# Global compiler instance
_compiler = None


def get_compiler() -> SparseCompiler:
    """Get global compiler instance."""
    global _compiler
    if _compiler is None:
        _compiler = SparseCompiler()
    return _compiler
