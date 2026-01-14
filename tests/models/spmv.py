# RUN: %PYTHON %s | FileCheck %s

"""
Test MVNet (sparse matrix-vector multiplication).
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import MVNet


def main():
    # CHECK-LABEL: pytorch
    # CHECK: tensor([ 385.,  935., 1485., 2035., 2585., 3135., 3685., 4235., 4785., 5335.])
    #
    # CHECK-LABEL: spardial
    # CHECK: [ 385.  935. 1485. 2035. 2585. 3135. 3685. 4235. 4785. 5335.]

    net = MVNet()

    # Get a fixed vector and matrix (which we make 2x2 block "sparse")
    dense_vector = torch.arange(1, 11, dtype=torch.float32)
    dense_input = torch.arange(1, 101, dtype=torch.float32).view(10, 10)
    sparse_matrix = dense_input.to_sparse_bsr(blocksize=(2, 2))

    print("pytorch")
    print(net(sparse_matrix, dense_vector))

    print("spardial")
    print(spardial_jit(net, sparse_matrix, dense_vector))


if __name__ == "__main__":
    main()
