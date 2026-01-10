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
cmake -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_EXTERNAL_PROJECTS='torch-mlir;spardial' \
      -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR='${PWD}/externals/torch-mlir' \
      -DLLVM_EXTERNAL_SPARDIAL_SOURCE_DIR='${PWD}' \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      externals/torch-mlir/externals/llvm-project/llvm
```