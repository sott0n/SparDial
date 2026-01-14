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
  - [LIT Tests (Integration Tests)](#lit-tests-integration-tests)
  - [Pytest Tests (Unit Tests)](#pytest-tests-unit-tests)

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

## How to test

SparDial has two types of tests: LIT integration tests and pytest unit tests.

### LIT Tests (Integration Tests)

Integration tests use LLVM's [LIT](https://llvm.org/docs/CommandGuide/lit.html) (LLVM Integrated Tester) framework to test the complete compilation pipeline with FileCheck assertions.

#### Running LIT tests

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

#### Test structure

LIT tests are located in `tests/models/` and test end-to-end compilation:
- **add.py**: Tests AddNet with dense and sparse tensors
  - Dense + Dense addition
  - Sparse (CSR) + Dense addition
  - Dense + Sparse (CSR) addition
  - Sparse + Sparse addition

Each test verifies output using FileCheck directives (`# CHECK:`) to ensure both PyTorch execution and SparDial JIT compilation produce expected results.

### Pytest Tests (Unit Tests)

Unit tests use pytest to test individual pipeline components.

#### Prerequisites

Install pytest if not already installed:

```shell
pip install pytest
```

#### Running pytest tests

```shell
# From the repository root
pytest tests/ -v
```

The test suite includes:
- **TestTorchDialectImport**: PyTorch model export to Torch Dialect IR using FxImporter
- **TestLinalgLowering**: Torch Dialect to Linalg-on-Tensors IR lowering
- **TestEndToEndPipeline**: Complete pipeline verification with parametrized tests

#### Pytest options

```shell
# Run specific test class
pytest tests/test_pipeline.py::TestTorchDialectImport -v

# Run specific test
pytest tests/test_pipeline.py::TestLinalgLowering::test_addnet_lowering -v

# Run with output capture disabled (see print statements)
pytest tests/ -v -s
```