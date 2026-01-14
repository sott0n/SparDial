"""SparDial JIT Compilation Backend"""

import sys
import torch
import numpy as np
import ctypes
from spardial.pipeline import (
    import_pytorch_model,
    lower_to_linalg,
    sparsify_and_bufferize,
    prepare_for_execution,
)
# Import MLIR ExecutionEngine and runtime utilities
try:
    from spardial.execution_engine import ExecutionEngine
except ImportError:
    # Fallback: Import directly from MLIR if spardial doesn't re-export it
    import spardial
    ExecutionEngine = spardial.execution_engine.ExecutionEngine

from spardial.runtime.np_to_memref import (
    get_unranked_memref_descriptor,
    unranked_memref_to_numpy,
    UnrankedMemRefDescriptor,
)


# Supported numpy dtypes for memref conversion
SUPPORTED_DTYPES = [
    np.float16,
    np.float32,
    np.float64,
    np.uint8,
    np.int8,
    np.int32,
    np.int64,
    np.bool_,
    np.complex64,
    np.complex128,
]

# Mapping from memref type strings to numpy dtypes
MEMREF_TYPE_TO_DTYPE = {
    "mrf16": np.float16,
    "mrf32": np.float32,
    "mrf64": np.float64,
    "mri1": np.bool_,
    "mri8": np.int8,
    "mri32": np.int32,
    "mri64": np.int64,
    "mrc32": np.complex64,
    "mrc64": np.complex128,
}

# Mapping from elemental types to ctypes
ELEMENTAL_TYPE_TO_CTYPE = {
    "i1": ctypes.c_bool,
    "i8": ctypes.c_byte,
    "i64": ctypes.c_int,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}

CONSUME_RETURN_FUNC_PREFIX = "refbackend_consume_func_return_"


def assert_dtype_supported(dtype):
    """Assert that the numpy dtype is supported for memref conversion."""
    assert dtype in SUPPORTED_DTYPES, (
        f"Only numpy arrays with dtypes in {SUPPORTED_DTYPES} are supported, "
        f"but got {dtype}"
    )


def get_return_funcs(module):
    """Extract return function names from the MLIR module."""
    return_funcs = []

    with module.context:
        for func in module.body:
            # Get function name
            func_name = str(func.attributes["sym_name"]).replace('"', "")
            if func_name.startswith(CONSUME_RETURN_FUNC_PREFIX):
                return_funcs.append(func_name)

    return return_funcs


def get_ctype_func(func_name):
    """
    Parse return function name to get ctypes function signature.

    Format: refbackend_consume_func_return_<type1>_<type2>_...
    """
    prefix_len = len(CONSUME_RETURN_FUNC_PREFIX)
    ret_types = func_name[prefix_len:].split("_")

    ctypes_args = [None]  # First arg is return type (None for void)
    for type_str in ret_types:
        if type_str in ELEMENTAL_TYPE_TO_CTYPE:
            ctypes_args.append(ELEMENTAL_TYPE_TO_CTYPE[type_str])
        elif type_str in MEMREF_TYPE_TO_DTYPE:
            # Pointer to UnrankedMemRefDescriptor
            ctypes_args.append(ctypes.POINTER(UnrankedMemRefDescriptor))
        else:
            raise ValueError(f"Unsupported type in return function: {type_str}")

    return ctypes.CFUNCTYPE(*ctypes_args), ret_types


class SparDialInvoker:
    """
    Wrapper for ExecutionEngine that handles tensor conversion and execution.
    """

    def __init__(self, module):
        """
        Initialize the invoker with an MLIR module.

        Args:
            module: MLIR Module ready for execution (after prepare_for_execution)
        """
        self.ee = ExecutionEngine(module)
        self.result = None

        # Register return value handlers
        return_funcs = get_return_funcs(module)

        for ret_func in return_funcs:
            ctype_wrapper, ret_types = get_ctype_func(ret_func)

            def consume_return_funcs(*args):
                # Convert arguments to numpy arrays
                results = []
                for arg, type_str in zip(args, ret_types):
                    if type_str in ELEMENTAL_TYPE_TO_CTYPE:
                        # Scalar value
                        results.append(arg)
                    else:
                        # Memref - convert to numpy
                        # arg is a POINTER(UnrankedMemRefDescriptor)
                        dtype = MEMREF_TYPE_TO_DTYPE[type_str]
                        # Cast arg to pointer and pass it
                        np_array = unranked_memref_to_numpy(arg, dtype)
                        results.append(np_array)

                self.result = tuple(results) if len(results) > 1 else results[0]

            self.ee.register_runtime(ret_func, ctype_wrapper(consume_return_funcs))

    def invoke(self, function_name, *args):
        """
        Invoke a function in the ExecutionEngine.

        Args:
            function_name: Name of the function to invoke
            *args: Numpy arrays as arguments

        Returns:
            Result numpy array(s)
        """
        # Convert numpy arrays to unranked memref descriptors
        ffi_args = []
        for arg in args:
            assert_dtype_supported(arg.dtype)
            # Create double pointer to unranked memref descriptor
            descriptor = get_unranked_memref_descriptor(arg)
            ffi_args.append(
                ctypes.pointer(ctypes.pointer(descriptor))
            )

        # Invoke the function
        self.ee.invoke(function_name, *ffi_args)

        # Get result
        result = self.result
        assert result is not None, "Invocation didn't produce a result"
        self.result = None

        return result


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

    # Get the function name from the module
    # Find the main function with a body (not extern/consume functions)
    function_name = None
    with exec_module.context:
        for func in exec_module.body:
            fname = str(func.attributes["sym_name"]).replace('"', "")
            # Skip refbackend consume functions
            if fname.startswith(CONSUME_RETURN_FUNC_PREFIX):
                continue
            # Check if function has a body (not an extern declaration)
            # Functions with bodies have regions
            if hasattr(func, 'regions') and len(func.regions) > 0:
                if len(func.regions[0].blocks) > 0:
                    function_name = fname
                    break

    if function_name is None:
        raise RuntimeError("No executable function found in module")

    print(f"Executing function: {function_name}", file=sys.stderr)

    # Execute
    result = invoker.invoke(function_name, *np_args)

    return result
