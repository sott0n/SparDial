# SparDial

Sparse Dialect Compiler based on MLIR

## Table of Contents

- [Pipeline](#pipeline)
- [How to build](#how-to-build)
- [Usage](#usage)
  - [JIT Compilation](#jit-compilation-with-spardial_jit)
  - [Sparse Tensor Support](#sparse-tensor-support)
  - [Available Models](#available-models)
- [How to test](#how-to-test)
- [Benchmarks](#benchmarks)

## Pipeline

The SparDial compilation pipeline transforms PyTorch models into optimized executable code through multiple MLIR dialect lowering stages:

```
PyTorch Model
    ↓ 1: torch.export.export() + FxImporter
Torch Dialect IR
    ↓ 2: torch-backend-to-linalg-on-tensors-backend-pipeline
Linalg-on-Tensors IR
    ↓ 3: sparsification-and-bufferization
Sparse Linalg IR (with bufferization)
    ↓ 4: convert-linalg-to-loops + lower to LLVM
LLVM Dialect IR
    ↓ 5: refback-munge-calling-conventions
Execution-ready IR
    ↓ 6: MLIR ExecutionEngine (JIT)
Executable Code
```

Key features:
- **Sparse tensor optimization at IR level**: Uses MLIR Sparse Tensor Dialect for compile-time optimization
- **Automatic sparsification**: Detects sparse patterns and applies optimizations during compilation
- **ExecutionEngine-based JIT**: Compiles and executes MLIR IR directly
- **CSR format support**: Handles Compressed Sparse Row tensors from PyTorch
- **End-to-end compilation**: PyTorch → MLIR → Optimized machine code

Note: The pipeline performs sparse optimizations at the IR level using Sparse Tensor Dialect. Currently, the execution interface uses dense array representation for input/output, while the internal computation benefits from sparse optimizations.

## How to build

Clone this repository and update submodules:

```shell
git clone https://github.com/sott0n/SparDial
cd SparDial
git submodule update --init --recursive --progress
```

Build SparDial as an in-tree project with LLVM/MLIR:

```shell
mkdir build && cd build

cmake -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_EXTERNAL_PROJECTS='torch-mlir;spardial' \
      -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="${PWD}/../externals/torch-mlir" \
      -DLLVM_EXTERNAL_SPARDIAL_SOURCE_DIR="${PWD}/.." \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      ../externals/torch-mlir/externals/llvm-project/llvm

# Build SparDial Python modules (includes FileCheck for testing)
ninja -j 32 SparDialPythonModules
```

This will build:
- SparDial Python bindings and modules
- MLIR/LLVM infrastructure
- FileCheck (for LIT tests)
- All required dependencies

## Usage

### JIT Compilation with spardial_jit

SparDial provides a JIT compilation function that compiles PyTorch models through the MLIR pipeline:

```python
import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import AddNet

# Create model and inputs
net = AddNet()
x = torch.arange(0, 16, dtype=torch.float32).view(4, 4)
y = torch.arange(16, 32, dtype=torch.float32).view(4, 4)

# JIT compile and execute
result = spardial_jit(net, x, y)
print(result)
```

### Sparse Tensor Support

SparDial supports sparse tensors in CSR (Compressed Sparse Row) format:

```python
import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import AddNet

# Create sparse tensor
sparse_matrix = torch.tensor([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0],
    [0.0, 0.0, 0.0, 0.0],
    [3.0, 0.0, 0.0, 0.0],
], dtype=torch.float32)
sparse_csr = sparse_matrix.to_sparse_csr()

# Dense tensor
dense = torch.arange(0, 16, dtype=torch.float32).view(4, 4)

# JIT compile with sparse tensor
net = AddNet()
result = spardial_jit(net, sparse_csr, dense)
print(result)

# Sparse + Sparse returns CSR components
crow_indices, col_indices, values = spardial_jit(net, sparse_csr, sparse_csr)
```

### NumPy/SciPy CSR SpMV (Decorator)

SparDial provides a direct MLIR path for SciPy CSR matrices without PyTorch,
via a `@spardial_jit`-decorated function:

```python
from scipy.sparse import csr_matrix
import numpy as np
from spardial import spardial_jit

@spardial_jit
def spmv(A, x):
    return A @ x

A_dense = np.array([
    [0, 0, 1, 0],
    [2, 0, 0, 0],
    [0, 3, 0, 4],
    [0, 0, 0, 0],
], dtype=np.float32)

A = csr_matrix(A_dense)
x = np.array([1, 2, 3, 4], dtype=np.float32)
y = spmv(A, x)
print(y)  # [ 3.  2. 22.  0.]
```

Notes:
- CSR format only (other sparse formats should be converted to CSR).
- Supported dtypes: float32, float64.
- Supported index dtypes: int32, int64.

## How to test

SparDial uses LLVM's [LIT](https://llvm.org/docs/CommandGuide/lit.html) (LLVM Integrated Tester) framework for all tests.

### Running tests

Using CMake:

```shell
# From the repository root
cmake --build build --target check-spardial
```

Or run LIT directly:

```shell
# From the build directory
cd build
python bin/llvm-lit -sv tools/spardial/tests
```

### Test structure

Tests are located in `tests/` directory:

- **tests/models/**: End-to-end model tests comparing PyTorch and SparDial JIT execution
  - `add.py`, `mul.py`, `mm.py`: Basic operations
  - `spmv.py`, `sddmm.py`: Sparse matrix operations
  - `gcn.py`, `gat.py`: Graph neural network layers
  - `sparse_formats.py`: Various sparse tensor formats (COO, CSR, CSC, BSR, BSC)

- **tests/pipeline/**: Pipeline stage tests
  - `torch_import.py`: PyTorch to Torch Dialect conversion
  - `linalg_lowering.py`: Torch Dialect to Linalg lowering
  - `sparse_encoding_pass.py`: Sparse encoding propagation pass
  - `sparsification.py`: Sparsification and bufferization
  - `sparse_csr_import.py`: CSR tensor automatic encoding

- **tests/numpy/**: NumPy/SciPy CSR backend tests
  - `test_spmv.py`: CSR SpMV correctness (float32/float64, empty, identity)

Each test uses FileCheck directives (`# CHECK:`) to verify expected output patterns.

## Benchmarks

SparDial includes a benchmarking system to compare performance between PyTorch CPU and SparDial JIT execution.

```shell
# Run quick benchmark
ninja spardial-benchmark-quick

# Run full benchmark
ninja spardial-benchmark

# Run with JSON output for CI
ninja spardial-benchmark-json
```

For detailed benchmark results and usage instructions, see [benchmarks/README.md](benchmarks/README.md).
