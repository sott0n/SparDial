"""PyTorch to MLIR Importer using FxImporter"""

import sys
import tempfile
import os
from io import StringIO
import torch
from spardial import ir
from spardial.dialects import torch as torch_d
from spardial.extras.fx_importer import FxImporter
from spardial.extras.fx_decomp_util import get_decomposition_table
from spardial.passmanager import PassManager


def import_pytorch_model(model, *example_args):
    """
    Convert a PyTorch model to an MLIR Torch Dialect module.

    Args:
        model: torch.nn.Module instance
        *example_args: example inputs for the model

    Returns:
        MLIR Module (Torch Dialect)
    """
    # 1. Create an MLIR Context
    context = ir.Context()

    # 2. Register the Torch Dialect
    torch_d.register_dialect(context)

    # 3. Create an FxImporter instance
    fx_importer = FxImporter(context=context)

    # 4. Export the PyTorch model to an FX graph
    print("Exporting PyTorch model to FX graph...", file=sys.stderr)
    prog = torch.export.export(model, example_args)

    # 5. Apply operator decompositions
    # Check if any arguments are sparse tensors
    has_sparse = any(
        hasattr(arg, "layout")
        and arg.layout
        in [
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]
        for arg in example_args
    )

    decomposition_table = get_decomposition_table()
    if decomposition_table and not has_sparse:
        # Skip decompositions for sparse tensors to avoid stride() errors
        print("Applying decompositions...", file=sys.stderr)
        prog = prog.run_decompositions(decomposition_table)
    elif has_sparse:
        print("Skipping decompositions for sparse tensors...", file=sys.stderr)

    # 6. Import to MLIR Torch Dialect
    print("Importing to MLIR Torch Dialect...", file=sys.stderr)
    fx_importer.import_frozen_program(prog)

    print("Import successful!", file=sys.stderr)
    return fx_importer.module


def print_mlir(module):
    print("\n" + "=" * 80)
    print("MLIR IR:")
    print("=" * 80)
    print(module)
    print("=" * 80 + "\n")


class SparDialCompilerError(Exception):
    """SparDial compilation error"""

    pass


def run_pipeline_with_repro_report(module, pipeline: str, description: str):
    """
    Run a pipeline and provide a detailed report on failure.

    Args:
        module: MLIR Module
        pipeline: Pipeline string
        description: Description for error message
    """
    original_stderr = sys.stderr
    try:
        sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True
        )

        with module.context:
            pm = PassManager.parse(pipeline)
            pm.run(module.operation)

    except Exception as e:
        module_name = "spardial_module"
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)

        message = f"""\
{description} failed with the following diagnostics:
{sys.stderr.getvalue()}

Python exception: {e}

The error can be reproduced with:
$ spardial-opt -pass-pipeline='{pipeline}' {filename}
"""
        raise SparDialCompilerError(message) from None
    finally:
        sys.stderr = original_stderr


def lower_to_linalg(torch_module):
    """
    Convert Torch Dialect IR to Linalg-on-Tensors IR

    Args:
        torch_module: MLIR Module (Torch Dialect)

    Returns:
        MLIR Module (Linalg-on-Tensors IR)
    """
    print("Lowering Torch Dialect -> Linalg-on-Tensors...", file=sys.stderr)

    pipeline = (
        "builtin.module("
        "func.func(torch-decompose-complex-ops),"
        "torch-backend-to-linalg-on-tensors-backend-pipeline"
        ")"
    )

    run_pipeline_with_repro_report(torch_module, pipeline, "Lowering Torch IR to Linalg IR")

    print("Lowering successful!", file=sys.stderr)
    return torch_module


def sparsify_and_bufferize(linalg_module, sparse_options=""):
    """
    Apply sparsification and bufferization passes to Linalg IR

    Args:
        linalg_module: MLIR Module with Linalg-on-Tensors IR
        sparse_options: Options for sparsification pass
                       (e.g., "parallelization-strategy=any-storage-any-loop")

    Returns:
        MLIR Module with sparse operations and bufferized memory
    """

    print("Applying sparsification and bufferization...", file=sys.stderr)

    # Build sparsification options string
    sp_opts = sparse_options if sparse_options else ""

    passes = [
        # Generalize named Linalg ops to generic form for sparsification
        "func.func(linalg-generalize-named-ops)",
        # Run pre-sparsification pass to fuse convert/cast op into
        # producer as they might hinder kernel fusions.
        "pre-sparsification-rewrite",
        # Fuse elementwise operations for better performance
        "func.func(linalg-fuse-elementwise-ops)",
        "convert-shape-to-std",
        # Propagate sparse encodings through operations (our custom pass)
        # TODO: Enable this pass once it's properly registered
        # "func.func(sparse-encoding-propagation)",
        # Configure sparse assembler for direct output
        "sparse-assembler{direct-out}",
        # Main sparsification and bufferization pass
        f"sparsification-and-bufferization{{{sp_opts}}}"
        if sp_opts
        else "sparsification-and-bufferization",
        # Convert sparse storage specifiers to LLVM
        "sparse-storage-specifier-to-llvm",
        # Expand realloc operations before bufferization
        "func.func(expand-realloc)",
        # Generalize pad and concat after sparse compiler, as they are handled
        # differently when the operations involve sparse operands.
        "func.func(refback-generalize-tensor-pad)",
        "func.func(refback-generalize-tensor-concat)",
        # Bufferize.
        "func.func(tm-tensor-bufferize)",
        "one-shot-bufferize{copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        "refback-mlprogram-bufferize",
        # Inline sparse helper methods where useful (but after dealloc).
        "inline",
        "refback-munge-calling-conventions",
        "func.func(tm-tensor-to-loops)",
        "func.func(refback-munge-memref-copy)",
        "func.func(convert-linalg-to-loops)",
        "func.func(lower-affine)",
        "convert-scf-to-cf",
        "func.func(refback-expand-ops-for-llvm)",
        "func.func(arith-expand)",
        "func.func(convert-math-to-llvm)",
        "convert-math-to-libm",
        "expand-strided-metadata",
        "finalize-memref-to-llvm",
        "lower-affine",
        "convert-bufferization-to-memref",
        "finalize-memref-to-llvm",
        "func.func(convert-arith-to-llvm)",
        # Vector code (SIMD):
        #   allow fp reductions to reassociate
        #   allow 32-bit index optimizations (unsafe for very large dimensions)
        #   assume we are running on a good ol' Intel X86 (disable for ARM/other)
        "convert-vector-to-llvm{reassociate-fp-reductions force-32bit-vector-indices enable-x86vector}",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "convert-complex-to-llvm",
        "reconcile-unrealized-casts",
    ]

    pipeline = "builtin.module(" + ",".join(passes) + ")"

    run_pipeline_with_repro_report(linalg_module, pipeline, "Sparsification and bufferization")

    return linalg_module


def prepare_for_execution(llvm_module):
    """
    Prepare LLVM module for execution by munging calling conventions.

    This makes the module compatible with ExecutionEngine by converting
    function signatures to use unranked memrefs and callback-based returns.

    Args:
        llvm_module: MLIR Module (LLVM Dialect)

    Returns:
        MLIR Module ready for ExecutionEngine
    """
    print("Preparing module for execution...", file=sys.stderr)

    passes = [
        # Munge calling conventions to make ExecutionEngine compatible
        # This converts function signatures to use unranked memrefs
        # and returns via callbacks
        "refback-munge-calling-conventions",
    ]

    pipeline = "builtin.module(" + ",".join(passes) + ")"

    run_pipeline_with_repro_report(llvm_module, pipeline, "Preparing for execution")

    print("Module ready for execution!", file=sys.stderr)
    return llvm_module
