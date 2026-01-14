# RUN: %PYTHON %s | FileCheck %s

"""
Test Normalization (D^-1 @ A @ D^-1) for graph processing.
"""

import torch
import numpy as np
from spardial.backend import spardial_jit
from spardial.models.kernels import Normalization


def main():
    # CHECK-LABEL: pytorch
    # CHECK: tensor({{\[}}[0.1111, 0.1667, 0.0000, 0.0000, 0.1667, 0.0000, 0.0000, 0.0000],
    # CHECK:         [0.0000, 0.2500, 0.0000, 0.0000, 0.2500, 0.0000, 0.0000, 0.0000],
    # CHECK:         [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    # CHECK:         [0.0000, 0.0000, 0.0000, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
    # CHECK:         [0.0000, 0.0000, 0.0000, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
    # CHECK:         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
    # CHECK:         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
    # CHECK:         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]{{\]}})
    #
    # CHECK-LABEL: spardial
    # CHECK: {{\[}}[0.
    # CHECK:  [0.   0.25 0.   0.   0.25 0.   0.   0.  ]
    # CHECK:  [0.   0.   1.   0.   0.   0.   0.   0.  ]
    # CHECK:  [0.   0.   0.   0.25 0.25 0.   0.   0.  ]
    # CHECK:  [0.   0.   0.   0.25 0.25 0.   0.   0.  ]
    # CHECK:  [0.   0.   0.   0.   0.   1.   0.   0.  ]
    # CHECK:  [0.   0.   0.   0.   0.   0.   1.   0.  ]
    # CHECK:  [0.   0.   0.   0.   0.   0.   0.   1.  ]{{\]}}

    net = Normalization()

    # Construct adjacency matrix
    V = 8
    edge = np.array([[0, 1], [0, 4], [1, 4], [3, 4], [4, 3]], dtype=np.int32)
    E = edge.shape[0]
    adj_mat = torch.sparse_coo_tensor(edge.T, torch.ones(E), (V, V), dtype=torch.int64)
    adj_mat = (
        torch.eye(V) + adj_mat
    )  # Add self-loops to the adjacency matrix (become dense)

    print("pytorch")
    print(net(adj_mat))

    print("spardial")
    print(spardial_jit(net, adj_mat))


if __name__ == "__main__":
    main()
