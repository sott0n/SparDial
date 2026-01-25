# RUN: %PYTHON %s | FileCheck %s

"""
Test various sparse tensor formats: COO, CSR, CSC, BSR, BSC.
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import AddNet


def print_coo(res):
    """Print COO format components: pos, indices..., values"""
    for component in res:
        print(component)


def print_csr_csc(res):
    """Print CSR/CSC format components: pos_indices, indices, values"""
    print(res[0])
    print(res[1])
    print(res[2])


def print_bsr_bsc(res):
    """Print BSR/BSC format components: pos_indices, indices, values"""
    print(res[0])
    print(res[1])
    print(res[2])


def test_coo():
    """Test COO (Coordinate) sparse format."""
    # CHECK-LABEL: === COO Format ===
    # CHECK: pytorch coo
    # CHECK: tensor(indices=tensor({{\[}}[0, 0, 1, 3],
    # CHECK:                              [1, 2, 3, 0]{{\]}}),
    # CHECK:        values=tensor([ 2.,  4.,  6., 10.]),
    # CHECK:        size=(4, 4), nnz=4, layout=torch.sparse_coo)
    # CHECK: spardial coo
    # CHECK: {{\[}}0 4{{\]}}
    # CHECK: {{\[}}0 0 1 3{{\]}}
    # CHECK: {{\[}}1 2 3 0{{\]}}
    # CHECK: {{\[}} 2.  4.  6. 10.{{\]}}

    print("=== COO Format ===")
    net = AddNet()

    # Create COO sparse tensor
    indices = torch.tensor([[0, 0, 1, 3], [1, 2, 3, 0]], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=torch.float32)
    S_coo = torch.sparse_coo_tensor(indices, values, size=(4, 4))

    print("pytorch coo")
    print(S_coo + S_coo)

    print("spardial coo")
    result = spardial_jit(net, S_coo, S_coo)
    print_coo(result)


def test_csr():
    """Test CSR (Compressed Sparse Row) format."""
    # CHECK-LABEL: === CSR Format ===
    # CHECK: pytorch csr
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}1, 3, 0{{\]}}),
    # CHECK:        values=tensor({{\[}}2., 4., 6.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csr)
    # CHECK: spardial csr
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}1 3 0{{\]}}
    # CHECK: {{\[}}2. 4. 6.{{\]}}

    print("=== CSR Format ===")
    net = AddNet()

    A = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    S_csr = A.to_sparse_csr()

    print("pytorch csr")
    print(S_csr + S_csr)

    print("spardial csr")
    result = spardial_jit(net, S_csr, S_csr)
    print_csr_csc(result)


def test_csc():
    """Test CSC (Compressed Sparse Column) format."""
    # CHECK-LABEL: === CSC Format ===
    # CHECK: pytorch csc
    # CHECK: tensor(ccol_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        row_indices=tensor({{\[}}3, 0, 1{{\]}}),
    # CHECK:        values=tensor({{\[}}6., 2., 4.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csc)
    # CHECK: spardial csc
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}3 0 1{{\]}}
    # CHECK: {{\[}}6. 2. 4.{{\]}}

    print("=== CSC Format ===")
    net = AddNet()

    A = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    S_csc = A.to_sparse_csc()

    print("pytorch csc")
    print(S_csc + S_csc)

    print("spardial csc")
    result = spardial_jit(net, S_csc, S_csc)
    print_csr_csc(result)


def test_bsr():
    """Test BSR (Block Sparse Row) format."""
    # CHECK-LABEL: === BSR Format ===
    # CHECK: pytorch bsr
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 2, 4{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}0, 1, 0, 1{{\]}}),
    # CHECK:        values=tensor({{\[}}{{\[}}{{\[}} 2.,  4.{{\]}}
    # CHECK: spardial bsr
    # CHECK: {{\[}}0 2 4{{\]}}
    # CHECK: {{\[}}0 1 0 1{{\]}}
    # CHECK: [ 2.  4.  6.  8.  6.  8. 10. 12. 10. 12. 14. 16. 14. 16. 18. 20.]

    print("=== BSR Format ===")
    net = AddNet()

    # Create a 4x4 matrix that can be blocked into 2x2 blocks
    A = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0, 8.0],
            [7.0, 8.0, 9.0, 10.0],
        ],
        dtype=torch.float32,
    )
    S_bsr = A.to_sparse_bsr(blocksize=(2, 2))

    print("pytorch bsr")
    print(S_bsr + S_bsr)

    print("spardial bsr")
    result = spardial_jit(net, S_bsr, S_bsr)
    print_bsr_bsc(result)


def test_bsc():
    """Test BSC (Block Sparse Column) format."""
    # CHECK-LABEL: === BSC Format ===
    # CHECK: pytorch bsc
    # CHECK: tensor(ccol_indices=tensor({{\[}}0, 2, 4{{\]}}),
    # CHECK:        row_indices=tensor({{\[}}0, 1, 0, 1{{\]}}),
    # CHECK:        values=tensor({{\[}}{{\[}}{{\[}} 2.,  4.{{\]}}
    # CHECK: spardial bsc
    # CHECK: {{\[}}0 2 4{{\]}}
    # CHECK: {{\[}}0 1 0 1{{\]}}
    # CHECK: [ 2.  4.  6.  8. 10. 12. 14. 16.  6.  8. 10. 12. 14. 16. 18. 20.]

    print("=== BSC Format ===")
    net = AddNet()

    # Create a 4x4 matrix that can be blocked into 2x2 blocks
    A = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0, 8.0],
            [7.0, 8.0, 9.0, 10.0],
        ],
        dtype=torch.float32,
    )
    S_bsc = A.to_sparse_bsc(blocksize=(2, 2))

    print("pytorch bsc")
    print(S_bsc + S_bsc)

    print("spardial bsc")
    result = spardial_jit(net, S_bsc, S_bsc)
    print_bsr_bsc(result)


def main():
    test_coo()
    test_csr()
    test_csc()
    test_bsr()
    test_bsc()


if __name__ == "__main__":
    main()
