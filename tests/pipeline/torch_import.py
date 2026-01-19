# RUN: %PYTHON %s | FileCheck %s

"""Test PyTorch to Torch Dialect IR conversion."""

import torch
from spardial.pipeline import import_pytorch_model
from spardial.models import AddNet, MulNet, SimpleLinear


# CHECK-LABEL: === test_addnet_import ===
print("=== test_addnet_import ===")
model = AddNet()
mlir_module = import_pytorch_model(model, torch.randn(2, 3), torch.randn(2, 3))
print(mlir_module)
# CHECK: func.func @main
# CHECK: torch.aten.add.Tensor
# CHECK: !torch.vtensor<[2,3],f32>


# CHECK-LABEL: === test_mulnet_import ===
print("=== test_mulnet_import ===")
model = MulNet()
mlir_module = import_pytorch_model(model, torch.randn(2, 3), torch.randn(2, 3))
print(mlir_module)
# CHECK: func.func @main
# CHECK: torch.aten.mul.Tensor
# CHECK: !torch.vtensor<[2,3],f32>


# CHECK-LABEL: === test_linear_import ===
print("=== test_linear_import ===")
model = SimpleLinear(in_features=4, out_features=2)
mlir_module = import_pytorch_model(model, torch.randn(1, 4))
print(mlir_module)
# Linear is decomposed to mm operation
# CHECK: func.func @main
# CHECK: !torch.vtensor<[1,4],f32>
# CHECK: torch.aten.mm
# CHECK: !torch.vtensor<[1,2],f32>
