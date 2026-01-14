# RUN: %PYTHON %s | FileCheck %s

"""
Test MMNet (matrix multiplication) with dense and sparse tensors.
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import MMNet


def print_sparse(res):
    print(res[0])
    print(res[1])
    print(res[2])


def main():
    # CHECK-LABEL: pytorch
    # CHECK: tensor({{\[}}[ 152.,  158.,  164.,  170.],
    # CHECK:         [ 504.,  526.,  548.,  570.],
    # CHECK:         [ 856.,  894.,  932.,  970.],
    # CHECK:         [1208., 1262., 1316., 1370.]{{\]}})
    # CHECK: tensor({{\[}}[20., 21., 22., 23.],
    # CHECK:         [56., 58., 60., 62.],
    # CHECK:         [ 0.,  0.,  0.,  0.],
    # CHECK:         [48., 51., 54., 57.]{{\]}})
    # CHECK: tensor({{\[}}[ 9.,  0.,  0.,  2.],
    # CHECK:         [21.,  4.,  0., 10.],
    # CHECK:         [33.,  8.,  0., 18.],
    # CHECK:         [45., 12.,  0., 26.]{{\]}})
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}3, 0, 1{{\]}}),
    # CHECK:        values=tensor({{\[}}2., 6., 3.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csr)
    #
    # CHECK-LABEL: spardial
    # CHECK: {{\[}}[ 152.  158.  164.  170.]
    # CHECK:  [ 504.  526.  548.  570.]
    # CHECK:  [ 856.  894.  932.  970.]
    # CHECK:  [1208. 1262. 1316. 1370.]{{\]}}
    # CHECK: {{\[}}[20. 21. 22. 23.]
    # CHECK:  [56. 58. 60. 62.]
    # CHECK:  [ 0.  0.  0.  0.]
    # CHECK:  [48. 51. 54. 57.]{{\]}}
    # CHECK: {{\[}}[ 9.  0.  0.  2.]
    # CHECK:  [21.  4.  0. 10.]
    # CHECK:  [33.  8.  0. 18.]
    # CHECK:  [45. 12.  0. 26.]{{\]}}
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}3 0 1{{\]}}
    # CHECK: {{\[}}2. 6. 3.{{\]}}

    net = MMNet()

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
    print(spardial_jit(net, S, Y))
    print(spardial_jit(net, X, S))
    print_sparse(spardial_jit(net, S, S))


if __name__ == "__main__":
    main()
