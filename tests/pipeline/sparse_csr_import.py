# RUN: %PYTHON %s | FileCheck %s

"""Test that sparse CSR tensor is imported with encoding annotation automatically."""

import torch
import spardial._mlir_libs._spardial
from spardial.pipeline import import_pytorch_model, lower_to_linalg, sparsify_and_bufferize
from spardial.models import AddNet


A = torch.tensor(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)
S = A.to_sparse_csr()

# Verify sparse properties
assert S.layout == torch.sparse_csr
assert S._nnz() == 3

Y = torch.arange(16, 32, dtype=torch.float32).view(4, 4)
net = AddNet()

# CHECK-LABEL: === torch_ir ===
print("=== torch_ir ===")
mlir_module = import_pytorch_model(net, S, Y)
torch_ir = str(mlir_module)
print(torch_ir)
# CHECK: sparse_tensor.encoding
# CHECK: compressed

# CHECK-LABEL: === linalg_ir ===
print("=== linalg_ir ===")
linalg_module = lower_to_linalg(mlir_module)
linalg_ir = str(linalg_module)
print(linalg_ir)
# CHECK: sparse_tensor.encoding

# CHECK-LABEL: === compiled_ir ===
print("=== compiled_ir ===")
compiled_module = sparsify_and_bufferize(linalg_module)
result_ir = str(compiled_module)
print(result_ir)
# CHECK: llvm.func @main
