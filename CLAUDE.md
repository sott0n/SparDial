# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparDial is a Sparse Dialect Compiler based on MLIR (Multi-Level Intermediate Representation). It compiles PyTorch models with sparse tensor support to optimized LLVM code.

## Build Commands

```shell
# Build the project
cd build && ninja

# Build Python modules only
ninja SparDialPythonModules
```

## Test Commands

```shell
# Run all tests (LIT-based)
ninja check-spardial

# Run LIT directly
python bin/llvm-lit -sv tools/spardial/tests
```

## Project Structure

- **python/spardial/**: Python package for PyTorch-to-MLIR pipeline
  - `pipeline.py`: Import and lowering functions
  - `backend.py`: JIT compilation backend (`spardial_jit`)
  - `models/`: Test model definitions (AddNet, MulNet, etc.)

- **lib/**: C++ MLIR dialect and passes
  - `Dialect/`: SparDial dialect definitions
  - `Transforms/`: Custom passes (e.g., sparse-encoding-propagation)

- **tests/**: LIT tests
  - `models/`: End-to-end model tests (PyTorch vs SparDial JIT)
  - `pipeline/`: Pipeline stage tests (import, lowering, sparsification)

## Compilation Pipeline

```
PyTorch Model
    ↓ torch.export.export() + FxImporter
Torch Dialect IR
    ↓ torch-backend-to-linalg-on-tensors-backend-pipeline
Linalg-on-Tensors IR
    ↓ sparsification-and-bufferization
LLVM Dialect IR
    ↓ ExecutionEngine JIT
Native Execution
```

## Key APIs

```python
from spardial.backend import spardial_jit
from spardial.pipeline import import_pytorch_model, lower_to_linalg, sparsify_and_bufferize
```
