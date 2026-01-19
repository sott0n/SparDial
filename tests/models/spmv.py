# RUN: %PYTHON %s | FileCheck %s

"""
Test MVNet (sparse matrix-vector multiplication) with various sparse formats.
Tests COO, CSR, CSC, BSR, and BSC formats.
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import MVNet


def test_coo():
    """Test SpMV with COO format."""
    # CHECK-LABEL: === COO SpMV ===
    # CHECK: pytorch coo
    # CHECK: tensor([ 385.,  935., 1485., 2035., 2585., 3135., 3685., 4235., 4785., 5335.])
    # CHECK: spardial coo
    # CHECK: [ 385.  935. 1485. 2035. 2585. 3135. 3685. 4235. 4785. 5335.]

    print("=== COO SpMV ===")
    net = MVNet()

    dense_vector = torch.arange(1, 11, dtype=torch.float32)
    dense_input = torch.arange(1, 101, dtype=torch.float32).view(10, 10)
    sparse_matrix = dense_input.to_sparse_coo()

    print("pytorch coo")
    print(net(sparse_matrix, dense_vector))

    print("spardial coo")
    print(spardial_jit(net, sparse_matrix, dense_vector))


def test_csr():
    """Test SpMV with CSR format."""
    # CHECK-LABEL: === CSR SpMV ===
    # CHECK: pytorch csr
    # CHECK: tensor([ 385.,  935., 1485., 2035., 2585., 3135., 3685., 4235., 4785., 5335.])
    # CHECK: spardial csr
    # CHECK: [ 385.  935. 1485. 2035. 2585. 3135. 3685. 4235. 4785. 5335.]

    print("=== CSR SpMV ===")
    net = MVNet()

    dense_vector = torch.arange(1, 11, dtype=torch.float32)
    dense_input = torch.arange(1, 101, dtype=torch.float32).view(10, 10)
    sparse_matrix = dense_input.to_sparse_csr()

    print("pytorch csr")
    print(net(sparse_matrix, dense_vector))

    print("spardial csr")
    print(spardial_jit(net, sparse_matrix, dense_vector))


def test_csc():
    """Test SpMV with CSC format."""
    # CHECK-LABEL: === CSC SpMV ===
    # CHECK: pytorch csc
    # CHECK: tensor([ 385.,  935., 1485., 2035., 2585., 3135., 3685., 4235., 4785., 5335.])
    # CHECK: spardial csc
    # CHECK: [ 385.  935. 1485. 2035. 2585. 3135. 3685. 4235. 4785. 5335.]

    print("=== CSC SpMV ===")
    net = MVNet()

    dense_vector = torch.arange(1, 11, dtype=torch.float32)
    dense_input = torch.arange(1, 101, dtype=torch.float32).view(10, 10)
    sparse_matrix = dense_input.to_sparse_csc()

    print("pytorch csc")
    print(net(sparse_matrix, dense_vector))

    print("spardial csc")
    print(spardial_jit(net, sparse_matrix, dense_vector))


def test_bsr():
    """Test SpMV with BSR format."""
    # CHECK-LABEL: === BSR SpMV ===
    # CHECK: pytorch bsr
    # CHECK: tensor([ 385.,  935., 1485., 2035., 2585., 3135., 3685., 4235., 4785., 5335.])
    # CHECK: spardial bsr
    # CHECK: [ 385.  935. 1485. 2035. 2585. 3135. 3685. 4235. 4785. 5335.]

    print("=== BSR SpMV ===")
    net = MVNet()

    dense_vector = torch.arange(1, 11, dtype=torch.float32)
    dense_input = torch.arange(1, 101, dtype=torch.float32).view(10, 10)
    sparse_matrix = dense_input.to_sparse_bsr(blocksize=(2, 2))

    print("pytorch bsr")
    print(net(sparse_matrix, dense_vector))

    print("spardial bsr")
    print(spardial_jit(net, sparse_matrix, dense_vector))


# NOTE: BSC SpMV is not supported by PyTorch (torch.mv does not support SparseBsc layout)
# def test_bsc(): ...


def main():
    test_coo()
    test_csr()
    test_csc()
    test_bsr()


if __name__ == "__main__":
    main()
