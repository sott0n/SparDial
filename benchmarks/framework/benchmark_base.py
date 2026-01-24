"""Base class and utilities for benchmarks."""

import io
import os
import sys
import time
from abc import ABC, abstractmethod
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from spardial.pipeline import (
    import_pytorch_model,
    lower_to_linalg,
    sparsify_and_bufferize,
    prepare_for_execution,
)
from spardial.backend import SparDialInvoker, CONSUME_RETURN_FUNC_PREFIX


@dataclass
class BenchmarkResult:
    """Container for benchmark timing results and statistics."""

    name: str
    size: Tuple[int, ...]
    sparsity: float
    format: str

    # PyTorch timing (in milliseconds)
    pytorch_times: List[float] = field(default_factory=list)
    pytorch_mean: float = 0.0
    pytorch_std: float = 0.0

    # SparDial timing (in milliseconds)
    spardial_compile_time: float = 0.0
    spardial_times: List[float] = field(default_factory=list)
    spardial_mean: float = 0.0
    spardial_std: float = 0.0

    # Comparison metrics
    speedup: float = 0.0

    # Correctness check
    correctness_passed: bool = False
    max_error: float = 0.0

    def compute_statistics(self) -> None:
        """Compute mean, std, and speedup from timing lists."""
        if self.pytorch_times:
            self.pytorch_mean = float(np.mean(self.pytorch_times))
            self.pytorch_std = float(np.std(self.pytorch_times))

        if self.spardial_times:
            self.spardial_mean = float(np.mean(self.spardial_times))
            self.spardial_std = float(np.std(self.spardial_times))

        if self.spardial_mean > 0 and self.pytorch_mean > 0:
            self.speedup = self.pytorch_mean / self.spardial_mean


class BenchmarkBase(ABC):
    """Abstract base class for benchmarks."""

    def __init__(
        self,
        name: str,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ):
        """
        Initialize benchmark.

        Args:
            name: Name of the benchmark
            warmup_iterations: Number of warmup runs before timing
            benchmark_iterations: Number of timed runs
        """
        self.name = name
        self.warmup_iterations = max(0, warmup_iterations)
        self.benchmark_iterations = max(1, benchmark_iterations)

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        """Create the PyTorch model to benchmark.

        Returns:
            PyTorch model instance
        """
        pass

    @abstractmethod
    def create_inputs(
        self,
        size: Tuple[int, ...],
        sparsity: float,
        format: str,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Create input tensors for the benchmark.

        Args:
            size: Size of the tensors
            sparsity: Sparsity level (0.0 to 1.0)
            format: Sparse format (coo, csr, csc, dense)

        Returns:
            Tuple of input tensors
        """
        pass

    def run_pytorch(
        self,
        model: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, float]:
        """
        Run PyTorch model and measure time.

        Args:
            model: PyTorch model
            inputs: Input tensors

        Returns:
            Tuple of (output tensor, execution time in ms)
        """
        # Synchronize before timing (if CUDA is available)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        output = model(*inputs)
        end = time.perf_counter()

        return output, (end - start) * 1000  # Convert to ms

    def compile_spardial(
        self,
        model: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
    ) -> Tuple[SparDialInvoker, str, float]:
        """
        Compile model using SparDial pipeline.

        Args:
            model: PyTorch model
            inputs: Example input tensors

        Returns:
            Tuple of (invoker, function_name, compile_time_ms)

        Raises:
            RuntimeError: If no executable function is found in the compiled module
        """
        start = time.perf_counter()

        # Import PyTorch model to MLIR
        mlir_module = import_pytorch_model(model, *inputs)

        # Lower Torch dialect to Linalg dialect
        linalg_module = lower_to_linalg(mlir_module)

        # Apply sparsification and bufferization
        llvm_module = sparsify_and_bufferize(linalg_module)

        # Prepare for execution
        exec_module = prepare_for_execution(llvm_module)

        # Create invoker
        invoker = SparDialInvoker(exec_module)

        # Find function name
        function_name: Optional[str] = None
        found_functions: List[str] = []
        with exec_module.context:
            for func in exec_module.body:
                fname = str(func.attributes["sym_name"]).replace('"', "")
                found_functions.append(fname)
                if fname.startswith(CONSUME_RETURN_FUNC_PREFIX):
                    continue
                if hasattr(func, "regions") and len(func.regions) > 0:
                    if len(func.regions[0].blocks) > 0:
                        function_name = fname
                        break

        if function_name is None:
            raise RuntimeError(
                f"No executable function found in module. "
                f"Found functions: {found_functions}"
            )

        end = time.perf_counter()

        return invoker, function_name, (end - start) * 1000

    def prepare_spardial_inputs(
        self,
        inputs: Tuple[torch.Tensor, ...],
    ) -> List[np.ndarray]:
        """
        Convert PyTorch tensors to numpy arrays for SparDial.

        Args:
            inputs: PyTorch tensors

        Returns:
            List of numpy arrays (sparse tensors are split into components)

        Raises:
            TypeError: If an unsupported argument type is provided
        """
        np_args: List[np.ndarray] = []
        for idx, arg in enumerate(inputs):
            if isinstance(arg, torch.Tensor):
                if arg.layout == torch.sparse_coo:
                    np_args.append(np.array([0, arg._nnz()], dtype=np.int64))
                    for dim_idx in arg._indices():
                        np_args.append(dim_idx.numpy())
                    np_args.append(arg._values().numpy())
                elif arg.layout == torch.sparse_csr or arg.layout == torch.sparse_bsr:
                    np_args.append(arg.crow_indices().numpy())
                    np_args.append(arg.col_indices().numpy())
                    np_args.append(arg.values().numpy())
                elif arg.layout == torch.sparse_csc or arg.layout == torch.sparse_bsc:
                    np_args.append(arg.ccol_indices().numpy())
                    np_args.append(arg.row_indices().numpy())
                    np_args.append(arg.values().numpy())
                else:
                    np_args.append(arg.detach().cpu().numpy())
            else:
                raise TypeError(
                    f"Unsupported argument type at index {idx}: {type(arg).__name__}. "
                    f"Expected torch.Tensor."
                )
        return np_args

    def run_spardial(
        self,
        invoker: SparDialInvoker,
        function_name: str,
        np_inputs: List[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """
        Run SparDial compiled model and measure time.

        Args:
            invoker: SparDialInvoker instance
            function_name: Name of the function to invoke
            np_inputs: Numpy arrays as inputs

        Returns:
            Tuple of (output array, execution time in ms)
        """
        start = time.perf_counter()
        output = invoker.invoke(function_name, *np_inputs)
        end = time.perf_counter()

        return output, (end - start) * 1000

    def _reconstruct_sparse_to_dense(
        self,
        sparse_components: Tuple[np.ndarray, ...],
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Reconstruct a dense array from sparse components.

        Args:
            sparse_components: Tuple of (indptr, indices, values) for CSR/CSC
            target_shape: Expected shape (rows, cols) of the output

        Returns:
            Dense numpy array
        """
        if len(sparse_components) != 3:
            raise ValueError(
                f"Expected 3 sparse components (indptr, indices, values), "
                f"got {len(sparse_components)}"
            )

        indptr, indices, values = sparse_components
        rows, cols = target_shape

        if HAS_SCIPY:
            # Use scipy for efficient reconstruction
            sparse_mat = csr_matrix(
                (values, indices, indptr),
                shape=(rows, cols)
            )
            return sparse_mat.toarray()
        else:
            # Fallback to pure numpy (slower but no dependency)
            result = np.zeros((rows, cols), dtype=values.dtype)
            for i in range(rows):
                for j in range(int(indptr[i]), int(indptr[i + 1])):
                    result[i, int(indices[j])] = values[j]
            return result

    def check_results(
        self,
        pytorch_output: Union[torch.Tensor, np.ndarray],
        spardial_output: Union[np.ndarray, Tuple[np.ndarray, ...]],
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> Tuple[bool, float]:
        """
        Check correctness between PyTorch and SparDial outputs.

        Args:
            pytorch_output: Output from PyTorch
            spardial_output: Output from SparDial (dense array or sparse components tuple)
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Tuple of (passed, max_error)
        """
        # Handle sparse outputs (SparDial returns tuple of components)
        if isinstance(spardial_output, tuple):
            # SparDial returned sparse tensor components
            if isinstance(pytorch_output, torch.Tensor) and pytorch_output.layout != torch.strided:
                # Both are sparse - compare the dense representations
                pytorch_np = pytorch_output.to_dense().detach().cpu().numpy()
                target_shape = pytorch_np.shape

                if len(spardial_output) == 3:
                    # CSR/CSC format - reconstruct to dense
                    try:
                        spardial_np = self._reconstruct_sparse_to_dense(
                            spardial_output, target_shape
                        )
                    except Exception:
                        # If reconstruction fails, compare values only
                        pytorch_values = pytorch_output._values().detach().cpu().numpy()
                        spardial_values = spardial_output[-1]
                        max_error = float(np.max(np.abs(
                            np.sort(pytorch_values) - np.sort(spardial_values)
                        )))
                        passed = bool(np.allclose(
                            np.sort(pytorch_values),
                            np.sort(spardial_values),
                            rtol=rtol,
                            atol=atol
                        ))
                        return passed, max_error
                else:
                    # COO format or other - compare values only
                    pytorch_values = pytorch_output._values().detach().cpu().numpy()
                    spardial_values = spardial_output[-1]
                    max_error = float(np.max(np.abs(
                        np.sort(pytorch_values) - np.sort(spardial_values)
                    )))
                    passed = bool(np.allclose(
                        np.sort(pytorch_values),
                        np.sort(spardial_values),
                        rtol=rtol,
                        atol=atol
                    ))
                    return passed, max_error
            else:
                # PyTorch is dense but SparDial returned sparse
                pytorch_np = (
                    pytorch_output.detach().cpu().numpy()
                    if isinstance(pytorch_output, torch.Tensor)
                    else np.array(pytorch_output)
                )
                spardial_np = spardial_output[-1]
                max_error = float('inf')
                passed = False
                return passed, max_error
        else:
            # Dense output - convert PyTorch output to numpy
            if isinstance(pytorch_output, torch.Tensor):
                if pytorch_output.layout != torch.strided:
                    pytorch_np = pytorch_output.to_dense().detach().cpu().numpy()
                else:
                    pytorch_np = pytorch_output.detach().cpu().numpy()
            elif isinstance(pytorch_output, np.ndarray):
                pytorch_np = pytorch_output
            else:
                pytorch_np = np.array(pytorch_output)

            spardial_np = spardial_output

        # Compute max absolute error
        max_error = float(np.max(np.abs(pytorch_np - spardial_np)))

        # Check if results are close
        passed = bool(np.allclose(pytorch_np, spardial_np, rtol=rtol, atol=atol))

        return passed, max_error

    def run(
        self,
        size: Tuple[int, ...],
        sparsity: float,
        format: str,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Run the complete benchmark.

        Args:
            size: Size of tensors
            sparsity: Sparsity level (0.0 to 1.0)
            format: Sparse format (coo, csr, csc, dense)
            verbose: Print detailed progress messages

        Returns:
            BenchmarkResult with timing and correctness info
        """
        result = BenchmarkResult(
            name=self.name,
            size=size,
            sparsity=sparsity,
            format=format,
        )

        # Create model and inputs
        model = self.create_model()
        inputs = self.create_inputs(size, sparsity, format)

        if verbose:
            print(f"Running benchmark: {self.name}", file=sys.stderr)
            print(f"  Size: {size}, Sparsity: {sparsity:.1%}, Format: {format}", file=sys.stderr)

        # PyTorch warmup
        if verbose:
            print("  PyTorch warmup...", file=sys.stderr)
        for _ in range(self.warmup_iterations):
            self.run_pytorch(model, inputs)

        # PyTorch benchmark
        if verbose:
            print("  PyTorch benchmark...", file=sys.stderr)
        pytorch_output = None
        for _ in range(self.benchmark_iterations):
            output, elapsed = self.run_pytorch(model, inputs)
            result.pytorch_times.append(elapsed)
            pytorch_output = output

        # SparDial compilation and execution
        spardial_output = None
        if verbose:
            # Run with stderr visible
            print("  SparDial compilation...", file=sys.stderr)
            invoker, function_name, compile_time = self.compile_spardial(model, inputs)
            result.spardial_compile_time = compile_time

            np_inputs = self.prepare_spardial_inputs(inputs)

            print("  SparDial warmup...", file=sys.stderr)
            for _ in range(self.warmup_iterations):
                self.run_spardial(invoker, function_name, np_inputs)

            print("  SparDial benchmark...", file=sys.stderr)
            for _ in range(self.benchmark_iterations):
                output, elapsed = self.run_spardial(invoker, function_name, np_inputs)
                result.spardial_times.append(elapsed)
                spardial_output = output
        else:
            # Suppress stderr during compilation using redirect_stderr
            with redirect_stderr(io.StringIO()):
                invoker, function_name, compile_time = self.compile_spardial(model, inputs)
                result.spardial_compile_time = compile_time

                np_inputs = self.prepare_spardial_inputs(inputs)

                for _ in range(self.warmup_iterations):
                    self.run_spardial(invoker, function_name, np_inputs)

                for _ in range(self.benchmark_iterations):
                    output, elapsed = self.run_spardial(invoker, function_name, np_inputs)
                    result.spardial_times.append(elapsed)
                    spardial_output = output

        # Check correctness
        if verbose:
            print("  Checking correctness...", file=sys.stderr)
        result.correctness_passed, result.max_error = self.check_results(
            pytorch_output, spardial_output
        )

        # Compute statistics
        result.compute_statistics()

        if verbose:
            status = "PASS" if result.correctness_passed else "FAIL"
            print(f"  Result: {status}, Speedup: {result.speedup:.2f}x", file=sys.stderr)

        return result
