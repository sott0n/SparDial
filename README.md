# SparDial
Sparse Dialect Compiler based on MLIR

## Pipeline
```
Pytorch Modle
    ↓ 1: torch.export.export() + FxImporter
Torch Dialect IR
    ↓ 2: torch-backend-to-linalg-on-tensors-backend-pipeline
Linalg-on-Tensors IR
    ↓ 3: sparse-encoding-propagation + spasification-and-bufferization
Sparse Linalg IR
    ↓ 4: convert-linalg-to-loop + bufferization
LLVM Dialect IR
    ↓ 5: ExecutionEngine
Executable Code via JIT
```

# How to build
Clone this repository and update submodule.

```shell
git clone https://github.com/sott0n/SparDial
cd SparDial
git submodule udpate --init --recursive --progress
```

Building the SparDial in-tree.

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

ninja -j 32 SparDialPythonModules
```

## How to test

### Prerequisites

Install pytest if not already installed:

```shell
pip install pytest
```

### Running tests

Run the test suite using pytest:

```shell
# From the repository root
pytest tests/ -v
```

The test suite includes:
- **TestTorchDialectImport**: PyTorch model export to Torch Dialect IR using FxImporter
- **TestLinalgLowering**: Torch Dialect to Linalg-on-Tensors IR lowering
- **TestEndToEndPipeline**: Complete pipeline verification with parametrized tests

### Test options

```shell
# Run specific test class
pytest tests/test_pipeline.py::TestTorchDialectImport -v

# Run specific test
pytest tests/test_pipeline.py::TestLinalgLowering::test_addnet_lowering -v

# Run with output capture disabled (see print statements)
pytest tests/ -v -s
```