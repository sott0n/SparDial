# RUN: %PYTHON %s | FileCheck %s

"""
Test MMNet (matrix multiplication) with various sparse formats: COO, CSR, CSC.
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import MMNet


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
    """Test matrix multiplication with COO format."""
    # CHECK-LABEL: === COO MM ===
    # CHECK: pytorch coo
    # CHECK: tensor(indices=tensor({{\[}}[0, 1, 3, 3],
    # CHECK:                              [3, 0, 1, 2]{{\]}}),
    # CHECK:        values=tensor([2., 6., 3., 6.]),
    # CHECK: spardial coo
    # CHECK: {{\[}}0 4{{\]}}
    # CHECK: {{\[}}0 1 3 3{{\]}}
    # CHECK: {{\[}}3 0 1 2{{\]}}
    # CHECK: {{\[}}2. 6. 3. 6.{{\]}}

    print("=== COO MM ===")
    net = MMNet()

    indices = torch.tensor([[0, 0, 1, 3], [1, 2, 3, 0]], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.float32)
    S = torch.sparse_coo_tensor(indices, values, size=(4, 4))

    print("pytorch coo")
    print(net(S, S))

    print("spardial coo")
    print_coo(spardial_jit(net, S, S))


def test_csr():
    """Test matrix multiplication with CSR format."""
    # CHECK-LABEL: === CSR MM ===
    # CHECK: pytorch csr
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}3, 0, 1{{\]}}),
    # CHECK:        values=tensor({{\[}}2., 6., 3.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csr)
    # CHECK: spardial csr
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}3 0 1{{\]}}
    # CHECK: {{\[}}2. 6. 3.{{\]}}

    print("=== CSR MM ===")
    net = MMNet()

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

    print("pytorch csr")
    print(net(S, S))

    print("spardial csr")
    print_csr_csc(spardial_jit(net, S, S))


def test_csc():
    """Test matrix multiplication with CSC format.

    Note: PyTorch's mm returns CSR format even for CSC input.
    """
    # CHECK-LABEL: === CSC MM ===
    # CHECK: pytorch csc
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}3, 0, 1{{\]}}),
    # CHECK:        values=tensor({{\[}}2., 6., 3.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csr)
    # CHECK: spardial csc
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}3 0 1{{\]}}
    # CHECK: {{\[}}2. 6. 3.{{\]}}

    print("=== CSC MM ===")
    net = MMNet()

    A = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    S = A.to_sparse_csc()

    print("pytorch csc")
    print(net(S, S))

    print("spardial csc")
    print_csr_csc(spardial_jit(net, S, S))


# NOTE: BSR and BSC mm is not supported by PyTorch (torch.mm fails with block sparse formats)


def main():
    test_coo()
    test_csr()
    test_csc()


if __name__ == "__main__":
    main()
