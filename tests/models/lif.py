# RUN: %PYTHON %s | FileCheck %s

"""
Test LIF (Leaky Integrate-and-Fire) neuron model.
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.lif import LIFSumOfSq


def main():
    # CHECK-LABEL: pytorch
    # CHECK: tensor([ 0., 11.,  9., 11., 13., 11., 10., 12.])
    #
    # CHECK-LABEL: spardial
    # CHECK: [ 0. 11.  9. 11. 13. 11. 10. 12.]

    net = LIFSumOfSq()

    # Get a random (but reproducible) input
    torch.manual_seed(0)
    x = torch.rand(2, 3, 8, 8)

    print("pytorch")
    print(net(x))

    print("spardial")
    print(spardial_jit(net, x))


if __name__ == "__main__":
    main()
