# RUN: %PYTHON %s | FileCheck %s

"""
Test GCN (Graph Convolutional Network).
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.gcn import graphconv_4_4, gcn_4_16_4


def main():
    # CHECK-LABEL: pytorch graphconv
    # CHECK: tensor({{\[}}[4.4778, 4.4778, 4.4778, 4.4778],
    # CHECK:         [5.7502, 5.7502, 5.7502, 5.7502],
    # CHECK:         [4.6980, 4.6980, 4.6980, 4.6980],
    # CHECK:         [3.6407, 3.6407, 3.6407, 3.6407]{{\]}})
    #
    # CHECK-LABEL: spardial graphconv
    # CHECK: {{\[}}[4.477828  4.477828  4.477828  4.477828 ]
    # CHECK:  [5.7501717 5.7501717 5.7501717 5.7501717]
    # CHECK:  [4.697952  4.697952  4.697952  4.697952 ]
    # CHECK:  [3.640687  3.640687  3.640687  3.640687 ]{{\]}}

    net = graphconv_4_4()
    net.eval()

    # Get random (but reproducible) matrices
    torch.manual_seed(0)
    inp = torch.rand(4, 4)
    adj_mat = torch.rand(4, 4).to_sparse()

    with torch.no_grad():
        print("pytorch graphconv")
        print(net(inp, adj_mat))

        print("spardial graphconv")
        print(spardial_jit(net, inp, adj_mat))

    # CHECK-LABEL: pytorch gcn
    # CHECK: tensor({{\[}}[-1.3863, -1.3863, -1.3863, -1.3863],
    # CHECK:         [-1.3863, -1.3863, -1.3863, -1.3863],
    # CHECK:         [-1.3863, -1.3863, -1.3863, -1.3863],
    # CHECK:         [-1.3863, -1.3863, -1.3863, -1.3863]{{\]}})
    #
    # CHECK-LABEL: spardial gcn
    # CHECK: {{\[}}[-1.3862944 -1.3862944 -1.3862944 -1.3862944]
    # CHECK:  [-1.3862944 -1.3862944 -1.3862944 -1.3862944]
    # CHECK:  [-1.3862944 -1.3862944 -1.3862944 -1.3862944]
    # CHECK:  [-1.3862944 -1.3862944 -1.3862944 -1.3862944]{{\]}}

    net2 = gcn_4_16_4()
    net2.eval()

    idx = torch.tensor([[0, 0, 1, 2], [0, 2, 3, 1]], dtype=torch.int64)
    val = torch.tensor([14.0, 3.0, -8.0, 11.0], dtype=torch.float32)
    S = torch.sparse_coo_tensor(idx, val, size=(4, 4))

    with torch.no_grad():
        print("pytorch gcn")
        print(net2(S, adj_mat))

        print("spardial gcn")
        print(spardial_jit(net2, S, adj_mat))


if __name__ == "__main__":
    main()
