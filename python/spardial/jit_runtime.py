"""Shared JIT runtime utilities for ExecutionEngine invocation."""

import ctypes
import numpy as np

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
    "i64": ctypes.c_int64,
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
            ctypes_args.append(ctypes.POINTER(UnrankedMemRefDescriptor))
        else:
            raise ValueError(f"Unsupported type in return function: {type_str}")

    return ctypes.CFUNCTYPE(*ctypes_args), ret_types


def find_executable_function(module):
    """Find the first non-extern, non-return-consumer function in a module."""
    with module.context:
        for func in module.body:
            fname = str(func.attributes["sym_name"]).replace('"', "")
            if fname.startswith(CONSUME_RETURN_FUNC_PREFIX):
                continue
            if hasattr(func, "regions") and len(func.regions) > 0:
                if len(func.regions[0].blocks) > 0:
                    return fname

    raise RuntimeError("No executable function found in module")


class SparDialInvoker:
    """Wrapper for ExecutionEngine that handles tensor conversion and execution."""

    def __init__(self, module, opt_level=None):
        """
        Initialize the invoker with an MLIR module.

        Args:
            module: MLIR Module ready for execution (after prepare_for_execution)
            opt_level: Optional optimization level for ExecutionEngine
        """
        if opt_level is None:
            self.ee = ExecutionEngine(module)
        else:
            self.ee = ExecutionEngine(module, opt_level=opt_level)
        self.result = None

        return_funcs = get_return_funcs(module)
        for ret_func in return_funcs:
            ctype_wrapper, ret_types = get_ctype_func(ret_func)

            def consume_return_funcs(*args):
                results = []
                for arg, type_str in zip(args, ret_types):
                    if type_str in ELEMENTAL_TYPE_TO_CTYPE:
                        results.append(arg)
                    else:
                        dtype = MEMREF_TYPE_TO_DTYPE[type_str]
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
        ffi_args = []
        for arg in args:
            assert_dtype_supported(arg.dtype)
            descriptor = get_unranked_memref_descriptor(arg)
            ffi_args.append(ctypes.pointer(ctypes.pointer(descriptor)))

        self.ee.invoke(function_name, *ffi_args)

        result = self.result
        assert result is not None, "Invocation didn't produce a result"
        self.result = None
        return result
