"""SparDial JIT Compilation Backend"""

import sys
import torch
import numpy as np
from spardial.pipeline import (
    import_pytorch_model,
    lower_to_linalg,
    sparsify_and_bufferize,
    prepare_for_execution,
)
from spardial.jit_runtime import SparDialInvoker, find_executable_function


def spardial_jit(model, *args, **kwargs):
    """
    JIT compile and execute a PyTorch model using SparDial MLIR pipeline.

    Args:
        model: torch.nn.Module instance or a NumPy/SciPy function (decorator form)
        *args: input tensors for the model

    Returns:
        Result tensor(s) from model execution
    """
    # NumPy/SciPy decorator path: @spardial_jit
    if callable(model) and not args and not kwargs and not isinstance(model, torch.nn.Module):
        def _wrapped(*fn_args, **fn_kwargs):
            from spardial.numpy_backend.tracer import trace_numpy
            op = trace_numpy(model, *fn_args, **fn_kwargs)
            return op.execute()

        return _wrapped

    if kwargs:
        raise TypeError("spardial_jit does not accept keyword arguments for PyTorch models.")

    # Import PyTorch model to MLIR
    mlir_module = import_pytorch_model(model, *args)

    # Lower Torch dialect to Linalg dialect
    linalg_module = lower_to_linalg(mlir_module)

    # Apply sparsification and bufferization
    llvm_module = sparsify_and_bufferize(linalg_module)

    # Prepare for execution (munge calling conventions)
    exec_module = prepare_for_execution(llvm_module)

    # Create invoker and execute
    invoker = SparDialInvoker(exec_module)

    # Convert PyTorch tensors to numpy arrays
    # Sparse tensors are split into their component arrays (CSR: crow_indices, col_indices, values)
    np_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if arg.layout == torch.sparse_coo:
                # COO format: position array + indices + values
                # Position array required by MLIR: [0, nnz]
                np_args.append(np.array([0, arg._nnz()], dtype=np.int64))
                # Transform indices: tensor<ndim x nnz> -> ndim x tensor<nnz>
                for idx in arg._indices():
                    np_args.append(idx.numpy())
                np_args.append(arg._values().numpy())
            elif arg.layout == torch.sparse_csr or arg.layout == torch.sparse_bsr:
                # CSR/BSR format: crow_indices, col_indices, values
                np_args.append(arg.crow_indices().numpy())
                np_args.append(arg.col_indices().numpy())
                np_args.append(arg.values().numpy())
            elif arg.layout == torch.sparse_csc or arg.layout == torch.sparse_bsc:
                # CSC/BSC format: ccol_indices, row_indices, values
                np_args.append(arg.ccol_indices().numpy())
                np_args.append(arg.row_indices().numpy())
                np_args.append(arg.values().numpy())
            else:
                # Dense tensor
                np_args.append(arg.detach().cpu().numpy())
        else:
            raise TypeError(f"Unsupported argument type: {type(arg)}")

    function_name = find_executable_function(exec_module)

    print(f"Executing function: {function_name}", file=sys.stderr)

    # Execute
    result = invoker.invoke(function_name, *np_args)

    return result
