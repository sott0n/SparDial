# RUN: %PYTHON %s | FileCheck %s

"""
Test AddNet with various combinations of dense and sparse tensors.
Tests both PyTorch native execution and SparDial JIT compilation.
"""

import torch
from spardial.backend import spardial_jit
from spardial.models.kernels import AddNet


def create_test_tensors():
    """Create test tensors for addition operations."""
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

    return X, Y, S


def run_pytorch_tests(net, X, Y, S):
    """Run tests with PyTorch native execution."""
    print("pytorch", torch.__version__)

    # Dense + Dense
    print(net(X, Y))

    # Sparse + Dense
    print(net(S, Y))

    # Dense + Sparse
    print(net(X, S))

    # Sparse + Sparse
    print(net(S, S))


def run_spardial_tests(net, X, Y, S):
    """Run tests with SparDial JIT compilation."""
    print("spardial")

    # Dense + Dense
    print(spardial_jit(net, X, Y))

    # Sparse + Dense
    print(spardial_jit(net, S, Y))

    # Dense + Sparse
    print(spardial_jit(net, X, S))

    # Sparse + Sparse (returns CSR components)
    crow, col, vals = spardial_jit(net, S, S)
    print(crow)
    print(col)
    print(vals)


def main():
    # CHECK-LABEL: pytorch
    # CHECK: tensor({{\[}}[16., 18., 20., 22.],
    # CHECK:         [24., 26., 28., 30.],
    # CHECK:         [32., 34., 36., 38.],
    # CHECK:         [40., 42., 44., 46.]{{\]}})
    # CHECK: tensor({{\[}}[16., 18., 18., 19.],
    # CHECK:         [20., 21., 22., 25.],
    # CHECK:         [24., 25., 26., 27.],
    # CHECK:         [31., 29., 30., 31.]{{\]}})
    # CHECK: tensor({{\[}}[ 0.,  2.,  2.,  3.],
    # CHECK:         [ 4.,  5.,  6.,  9.],
    # CHECK:         [ 8.,  9., 10., 11.],
    # CHECK:         [15., 13., 14., 15.]{{\]}})
    # CHECK: tensor(crow_indices=tensor({{\[}}0, 1, 2, 2, 3{{\]}}),
    # CHECK:        col_indices=tensor({{\[}}1, 3, 0{{\]}}),
    # CHECK:        values=tensor({{\[}}2., 4., 6.{{\]}}), size=(4, 4), nnz=3,
    # CHECK:        layout=torch.sparse_csr)
    #
    # CHECK-LABEL: spardial
    # CHECK: {{\[}}[16. 18. 20. 22.]
    # CHECK:  [24. 26. 28. 30.]
    # CHECK:  [32. 34. 36. 38.]
    # CHECK:  [40. 42. 44. 46.]{{\]}}
    # CHECK: {{\[}}[16. 18. 18. 19.]
    # CHECK:  [20. 21. 22. 25.]
    # CHECK:  [24. 25. 26. 27.]
    # CHECK:  [31. 29. 30. 31.]{{\]}}
    # CHECK: {{\[}}[ 0.  2.  2.  3.]
    # CHECK:  [ 4.  5.  6.  9.]
    # CHECK:  [ 8.  9. 10. 11.]
    # CHECK:  [15. 13. 14. 15.]{{\]}}
    # CHECK: {{\[}}0 1 2 2 3{{\]}}
    # CHECK: {{\[}}1 3 0{{\]}}
    # CHECK: {{\[}}2. 4. 6.{{\]}}

    net = AddNet()
    X, Y, S = create_test_tensors()

    run_pytorch_tests(net, X, Y, S)
    run_spardial_tests(net, X, Y, S)


if __name__ == "__main__":
    main()
