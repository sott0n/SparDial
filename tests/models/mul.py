# RUN: %PYTHON %s | FileCheck %s

"""
Test MulNet with various sparse tensor formats: COO, CSR, CSC, BSR, BSC.
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import MulNet


def print_coo(res):
    """Print COO format components."""
    for component in res:
        print(component)


def print_csr_csc(res):
    """Print CSR/CSC format components."""
    print(res[0])
    print(res[1])
    print(res[2])




def test_coo():
    """Test element-wise multiplication with COO format."""
    # CHECK-LABEL: === COO Mul ===
    # CHECK: pytorch coo
    # CHECK: tensor(indices=tensor({{\[}}[0, 0, 1, 3],
    # CHECK:                              [1, 2, 3, 0]{{\]}}),
    # CHECK:        values=tensor([1., 4., 9., 9.]),
    # CHECK: spardial coo
    # CHECK: {{\[}}0 4{{\]}}
    # CHECK: {{\[}}0 0 1 3{{\]}}
    # CHECK: {{\[}}1 2 3 0{{\]}}
    # CHECK: {{\[}}1. 4. 9. 9.{{\]}}

    print("=== COO Mul ===")
    net = MulNet()

    indices = torch.tensor([[0, 0, 1, 3], [1, 2, 3, 0]], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0, 3.0, 3.0], dtype=torch.float32)
    S = torch.sparse_coo_tensor(indices, values, size=(4, 4))

    print("pytorch coo")
    print(net(S, S))

    print("spardial coo")
    print_coo(spardial_jit(net, S, S))


def test_csr():
    """Test element-wise multiplication with CSR format."""
    # CHECK-LABEL: === CSR Mul ===
    # CHECK: pytorch csr
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}1, 3, 0{{\]}}),
    # CHECK:        values=tensor({{\[}}1., 4., 9.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csr)
    # CHECK: spardial csr
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}1 3 0{{\]}}
    # CHECK: {{\[}}1. 4. 9.{{\]}}

    print("=== CSR Mul ===")
    net = MulNet()

    A = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)
    S = A.to_sparse_csr()

    print("pytorch csr")
    print(net(S, S))

    print("spardial csr")
    print_csr_csc(spardial_jit(net, S, S))


# NOTE: CSC, BSR, BSC mul is not supported by PyTorch (torch.export fails)
# PyTorch's torch.mul only supports COO and CSR formats for sparse tensors


def main():
    test_coo()
    test_csr()


if __name__ == "__main__":
    main()
