# RUN: %PYTHON %s | FileCheck %s

"""Test Torch Dialect to Linalg-on-Tensors IR conversion."""

import torch
from spardial.pipeline import import_pytorch_model, lower_to_linalg
from spardial.models import AddNet, MulNet


# CHECK-LABEL: === test_addnet_lowering ===
print("=== test_addnet_lowering ===")
model = AddNet()
mlir_module = import_pytorch_model(model, torch.randn(2, 3), torch.randn(2, 3))
linalg_module = lower_to_linalg(mlir_module)
print(linalg_module)
# CHECK: affine_map
# CHECK: linalg.generic
# CHECK: tensor<2x3xf32>
# CHECK: arith.addf
# CHECK-NOT: torch.aten
# CHECK-NOT: !torch.vtensor


# CHECK-LABEL: === test_mulnet_lowering ===
print("=== test_mulnet_lowering ===")
model = MulNet()
mlir_module = import_pytorch_model(model, torch.randn(2, 3), torch.randn(2, 3))
linalg_module = lower_to_linalg(mlir_module)
print(linalg_module)
# CHECK: affine_map
# CHECK: linalg.generic
# CHECK: tensor<2x3xf32>
# CHECK: arith.mulf
# CHECK-NOT: torch.aten
