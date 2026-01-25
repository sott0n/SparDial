# RUN: %PYTHON %s | FileCheck %s

"""Test sparsification and bufferization pipeline."""

import torch
import spardial._mlir_libs._spardial
from spardial.pipeline import import_pytorch_model, lower_to_linalg, sparsify_and_bufferize
from spardial.models import AddNet, MulNet


# CHECK-LABEL: === test_sparsify_basic ===
print("=== test_sparsify_basic ===")
model = AddNet()
mlir_module = import_pytorch_model(model, torch.randn(2, 3), torch.randn(2, 3))
linalg_module = lower_to_linalg(mlir_module)

# Verify input is tensor-based
ir_before = str(linalg_module)
assert "tensor<2x3xf32>" in ir_before

compiled_module = sparsify_and_bufferize(linalg_module)
print(compiled_module)
# CHECK: llvm.func @main


# CHECK-LABEL: === test_sparsify_with_options ===
print("=== test_sparsify_with_options ===")
model = MulNet()
mlir_module = import_pytorch_model(model, torch.randn(3, 3), torch.randn(3, 3))
linalg_module = lower_to_linalg(mlir_module)

compiled_module = sparsify_and_bufferize(
    linalg_module, sparse_options="parallelization-strategy=none"
)
print(compiled_module)
# CHECK: llvm.func @main
