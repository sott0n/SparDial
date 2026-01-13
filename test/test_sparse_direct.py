import torch
import pytest

from spardial.pipeline import import_pytorch_model, lower_to_linalg, sparsify_and_bufferize
from spardial.models import AddNet


def test_sparse_csr_direct_import():
    """Test that sparse CSR tensor is imported with encoding annotation automatically"""
    import spardial._mlir_libs._spardial

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

    mlir_module = import_pytorch_model(net, S, Y)
    assert mlir_module is not None

    torch_ir = str(mlir_module)

    assert 'sparse_tensor.encoding' in torch_ir, \
        "FxImporter should automatically add sparse encoding for PyTorch sparse tensors"
    assert 'compressed' in torch_ir, \
        "CSR encoding should contain 'compressed' level type"

    linalg_module = lower_to_linalg(mlir_module)
    assert linalg_module is not None

    linalg_ir = str(linalg_module)
    assert 'sparse_tensor.encoding' in linalg_ir, \
        "Sparse encoding should be preserved after lowering to Linalg"

    compiled_module = sparsify_and_bufferize(linalg_module)
    assert compiled_module is not None

    result_ir = str(compiled_module)
    assert 'llvm.func @main' in result_ir
