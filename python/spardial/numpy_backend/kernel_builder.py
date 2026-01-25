"""Sparse operation kernel MLIR module construction (SpMV MVP)."""

from typing import List
from enum import Enum, auto
import numpy as np

from spardial import ir
from spardial.dialects import func
from spardial.dialects.linalg.opdsl import lang as dsl

from .input_spec import InputSpec
from .sparse_encoding import SparseEncodingBuilder


class KernelType(Enum):
    """Supported kernel types."""

    SPMV = auto()  # Sparse Matrix-Vector: y = A @ x
    SPMM = auto()  # Sparse Matrix-Matrix: C = A @ B (A sparse, B dense)
    # Future: SDDMM, etc.


@dsl.linalg_structured_op
def matvec_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N),
    x=dsl.TensorDef(dsl.T, dsl.S.N),
    y=dsl.TensorDef(dsl.T, dsl.S.M, output=True),
):
    """SpMV: y[m] += A[m, n] * x[n]"""
    y[dsl.D.m] += A[dsl.D.m, dsl.D.n] * x[dsl.D.n]


@dsl.linalg_structured_op
def matmul_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N),
    B=dsl.TensorDef(dsl.T, dsl.S.N, dsl.S.K),
    C=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K, output=True),
):
    """SpMM: C[m, k] += A[m, n] * B[n, k]"""
    C[dsl.D.m, dsl.D.k] += A[dsl.D.m, dsl.D.n] * B[dsl.D.n, dsl.D.k]


class KernelBuilder:
    """Builder for sparse operation kernels."""

    def __init__(self):
        self._kernel_builders = {
            KernelType.SPMV: self._build_spmv,
            KernelType.SPMM: self._build_spmm,
        }

    def build(
        self,
        kernel_type: KernelType,
        input_specs: List[InputSpec],
        output_spec: InputSpec,
    ) -> ir.Module:
        """Build MLIR module from kernel type and input/output specifications.

        Args:
            kernel_type: Type of kernel to build
            input_specs: List of input tensor specifications
            output_spec: Output tensor specification

        Returns:
            MLIR Module containing the kernel function
        """
        if kernel_type not in self._kernel_builders:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        return self._kernel_builders[kernel_type](input_specs, output_spec)

    def _build_spmv(
        self,
        input_specs: List[InputSpec],
        output_spec: InputSpec,
    ) -> ir.Module:
        """Build SpMV kernel: y = A @ x

        Args:
            input_specs: [A_spec, x_spec] where A is sparse and x is dense
            output_spec: y_spec (dense vector)

        Returns:
            MLIR Module with SpMV function
        """
        assert len(input_specs) == 2, "SpMV requires 2 inputs: A (sparse), x (dense)"

        A_spec, x_spec = input_specs

        context = ir.Context()
        encoding_builder = SparseEncodingBuilder(context)

        with context, ir.Location.unknown():
            # Element type
            if A_spec.dtype == np.float32:
                elem_type = ir.F32Type.get()
            else:
                elem_type = ir.F64Type.get()

            # Sparse encoding for A (CSR)
            sparse_encoding = encoding_builder.build(A_spec.format, A_spec.index_dtype or np.int64)

            # Types
            A_type = ir.RankedTensorType.get(list(A_spec.shape), elem_type, sparse_encoding)
            x_type = ir.RankedTensorType.get(list(x_spec.shape), elem_type)
            y_type = ir.RankedTensorType.get(list(output_spec.shape), elem_type)

            # Build module
            module = ir.Module.create()

            with ir.InsertionPoint(module.body):

                @func.FuncOp.from_py_func(A_type, x_type, y_type)
                def spmv(A, x, y):
                    return matvec_dsl(A, x, outs=[y])

            # Add llvm.emit_c_interface attribute to generate C-compatible wrapper
            for op in module.body:
                if hasattr(op, "attributes") and "sym_name" in op.attributes:
                    name = str(op.attributes["sym_name"]).strip('"')
                    if name == "spmv":
                        op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

            return module

    def _build_spmm(
        self,
        input_specs: List[InputSpec],
        output_spec: InputSpec,
    ) -> ir.Module:
        """Build SpMM kernel: C = A @ B

        Args:
            input_specs: [A_spec, B_spec] where A is sparse and B is dense
            output_spec: C_spec (dense matrix)

        Returns:
            MLIR Module with SpMM function
        """
        assert len(input_specs) == 2, "SpMM requires 2 inputs: A (sparse), B (dense)"

        A_spec, B_spec = input_specs

        context = ir.Context()
        encoding_builder = SparseEncodingBuilder(context)

        with context, ir.Location.unknown():
            # Element type
            if A_spec.dtype == np.float32:
                elem_type = ir.F32Type.get()
            else:
                elem_type = ir.F64Type.get()

            # Sparse encoding for A (CSR)
            sparse_encoding = encoding_builder.build(A_spec.format, A_spec.index_dtype or np.int64)

            # Types
            A_type = ir.RankedTensorType.get(list(A_spec.shape), elem_type, sparse_encoding)
            B_type = ir.RankedTensorType.get(list(B_spec.shape), elem_type)
            C_type = ir.RankedTensorType.get(list(output_spec.shape), elem_type)

            # Build module
            module = ir.Module.create()

            with ir.InsertionPoint(module.body):

                @func.FuncOp.from_py_func(A_type, B_type, C_type)
                def spmm(A, B, C):
                    return matmul_dsl(A, B, outs=[C])

            # Add llvm.emit_c_interface attribute to generate C-compatible wrapper
            for op in module.body:
                if hasattr(op, "attributes") and "sym_name" in op.attributes:
                    name = str(op.attributes["sym_name"]).strip('"')
                    if name == "spmm":
                        op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

            return module
