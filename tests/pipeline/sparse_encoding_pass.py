# RUN: %PYTHON %s | FileCheck %s

"""Test custom SparDial sparse-encoding-propagation pass."""

import torch
import spardial._mlir_libs._spardial
from spardial.pipeline import import_pytorch_model, lower_to_linalg
from spardial.passmanager import PassManager
from spardial.models import AddNet, MulNet


# CHECK-LABEL: === test_sparse_encoding_propagation_addnet ===
print("=== test_sparse_encoding_propagation_addnet ===")
model = AddNet()
mlir_module = import_pytorch_model(model, torch.randn(2, 3), torch.randn(2, 3))
linalg_module = lower_to_linalg(mlir_module)

with linalg_module.context:
    pm = PassManager.parse("builtin.module(func.func(sparse-encoding-propagation))")
    pm.run(linalg_module.operation)

print(linalg_module)
# CHECK: func.func @main
# CHECK: linalg.generic


# CHECK-LABEL: === test_sparse_encoding_propagation_mulnet ===
print("=== test_sparse_encoding_propagation_mulnet ===")
model = MulNet()
mlir_module = import_pytorch_model(model, torch.randn(2, 3), torch.randn(2, 3))
linalg_module = lower_to_linalg(mlir_module)

with linalg_module.context:
    pm = PassManager.parse("builtin.module(func.func(sparse-encoding-propagation))")
    pm.run(linalg_module.operation)

print(linalg_module)
# CHECK: func.func @main
# CHECK: linalg.generic
