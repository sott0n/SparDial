"""SparDial JIT Compilation Backend"""

import torch
import numpy as np
import ctypes
from spardial.pipeline import (
    import_pytorch_model,
    lower_to_linalg,
    sparsify_and_bufferize,
)


def spardial_jit(model, *args):
    """
    JIT compile and execute a PyTorch model using SparDial MLIR pipeline.

    Args:
        model: torch.nn.Module instance
        *args: input tensors for the model

    Returns:
        Result tensor(s) from model execution
    """
    # Import PyTorch model to MLIR
    mlir_module = import_pytorch_model(model, *args)

    # Lower Torch dialect to Linalg dialect
    linalg_module = lower_to_linalg(mlir_module)

    # Apply sparsification and bufferization
    llvm_module = sparsify_and_bufferize(linalg_module)

    # TODO: Execute using MLIR ExecutionEngine
    # For now, we'll use PyTorch execution as a placeholder
    # This will be replaced with actual MLIR execution

    # Execute the model using PyTorch
    with torch.no_grad():
        result = model(*args)

    # Handle sparse tensor output
    if hasattr(result, 'layout') and result.layout == torch.sparse_csr:
        # Return CSR components as a tuple for sparse tensors
        crow_indices = result.crow_indices().detach().cpu().numpy()
        col_indices = result.col_indices().detach().cpu().numpy()
        values = result.values().detach().cpu().numpy()
        return (crow_indices, col_indices, values)

    # Return the result as a numpy array for dense tensors
    if isinstance(result, torch.Tensor):
        return result.detach().cpu().numpy()

    return result
