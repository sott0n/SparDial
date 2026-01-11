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
    print("Exporting PyTorch model to FX graph...")
    prog = torch.export.export(model, example_args)

    # 5. Apply operator decompositions
    # Check if any arguments are sparse tensors
    has_sparse = any(
        hasattr(arg, 'layout') and arg.layout in [
            torch.sparse_coo, torch.sparse_csr, torch.sparse_csc,
            torch.sparse_bsr, torch.sparse_bsc
        ]
        for arg in example_args
    )

    decomposition_table = get_decomposition_table()
    if decomposition_table and not has_sparse:
        # Skip decompositions for sparse tensors to avoid stride() errors
        print("Applying decompositions...")
        prog = prog.run_decompositions(decomposition_table)
    elif has_sparse:
        print("Skipping decompositions for sparse tensors...")

    # 6. Import to MLIR Torch Dialect
    print("Importing to MLIR Torch Dialect...")
    fx_importer.import_frozen_program(prog)

    print("Import successful!")
    return fx_importer.module


def print_mlir(module):
    print("\n" + "="*80)
    print("MLIR IR:")
    print("="*80)
    print(module)
    print("="*80 + "\n")


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
    print("Lowering Torch Dialect -> Linalg-on-Tensors...")

    pipeline = (
        "builtin.module("
        "func.func(torch-decompose-complex-ops),"
        "torch-backend-to-linalg-on-tensors-backend-pipeline"
        ")"
    )

    run_pipeline_with_repro_report(
        torch_module,
        pipeline,
        "Lowering Torch IR to Linalg IR"
    )

    print("Lowering successful!")
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

    print("Applying sparsification and bufferization...")

    # Build sparsification options string
    sp_opts = sparse_options if sparse_options else ""

    passes = [
        # Generalize named Linalg ops to generic form for sparsification
        "func.func(linalg-generalize-named-ops)",

        # Fuse elementwise operations for better performance
        "func.func(linalg-fuse-elementwise-ops)",

        # Propagate sparse encodings through operations (our custom pass)
        "func.func(sparse-encoding-propagation)",

        # Configure sparse assembler for direct output
        "sparse-assembler{direct-out}",

        # Main sparsification and bufferization pass
        f"sparsification-and-bufferization{{{sp_opts}}}" if sp_opts
        else "sparsification-and-bufferization",

        # Convert sparse storage specifiers to LLVM
        "sparse-storage-specifier-to-llvm",

        # Expand realloc operations before bufferization
        "func.func(expand-realloc)",

        # One-shot bufferization: convert tensors to buffers
        "one-shot-bufferize{"
        "copy-before-write "
        "bufferize-function-boundaries "
        "function-boundary-type-conversion=identity-layout-map"
        "}",

        # Convert remaining tensor operations to buffers
        "func.func(buffer-deallocation-simplification)",
        "func.func(convert-linalg-to-affine-loops)",
    ]

    pipeline = "builtin.module(" + ",".join(passes) + ")"

    run_pipeline_with_repro_report(
        linalg_module,
        pipeline,
        "Sparsification and bufferization"
    )

    print("Sparsification and bufferization successful!")
    return linalg_module
