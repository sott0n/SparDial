# RUN: %PYTHON %s | FileCheck %s

"""
Test MulNet with various combinations of dense and sparse tensors.
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import MulNet


def print_sparse(res):
    print(res[0])
    print(res[1])
    print(res[2])


def main():
    # CHECK-LABEL: pytorch
    # CHECK: tensor({{\[}}[  0.,  17.,  36.,  57.],
    # CHECK:         [ 80., 105., 132., 161.],
    # CHECK:         [192., 225., 260., 297.],
    # CHECK:         [336., 377., 420., 465.]{{\]}})
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}1, 3, 0{{\]}}),
    # CHECK:        values=tensor({{\[}}17., 46., 84.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csr)
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}1, 3, 0{{\]}}),
    # CHECK:        values=tensor({{\[}} 1., 14., 36.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csr)
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}1, 3, 0{{\]}}),
    # CHECK:        values=tensor({{\[}}1., 4., 9.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csr)
    #
    # CHECK-LABEL: spardial
    # CHECK: {{\[}}[  0.  17.  36.  57.]
    # CHECK:  [ 80. 105. 132. 161.]
    # CHECK:  [192. 225. 260. 297.]
    # CHECK:  [336. 377. 420. 465.]{{\]}}
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}1 3 0{{\]}}
    # CHECK: {{\[}}17. 46. 84.{{\]}}
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}1 3 0{{\]}}
    # CHECK: {{\[}} 1. 14. 36.{{\]}}
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}1 3 0{{\]}}
    # CHECK: {{\[}}1. 4. 9.{{\]}}

    net = MulNet()

    # Dense tensors
    X = torch.arange(0, 16, dtype=torch.float32).view(4, 4)
    Y = torch.arange(16, 32, dtype=torch.float32).view(4, 4)

    # Sparse tensor (CSR format)
    A = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)
    S = A.to_sparse_csr()

    print("pytorch")
    print(net(X, Y))
    print(net(S, Y))
    print(net(X, S))
    print(net(S, S))

    print("spardial")
    print(spardial_jit(net, X, Y))
    print_sparse(spardial_jit(net, S, Y))
    print_sparse(spardial_jit(net, X, S))
    print_sparse(spardial_jit(net, S, S))


if __name__ == "__main__":
    main()
