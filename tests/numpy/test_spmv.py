# RUN: %PYTHON %s | FileCheck %s

"""
Test SpMV with NumPy/SciPy backend.
Tests basic SpMV, identity matrix, empty matrix, and column vector input.
"""

from scipy.sparse import csr_matrix
import numpy as np
from spardial import spmv


def test_spmv_basic():
    """Basic SpMV test"""
    # CHECK-LABEL: === Basic SpMV ===
    print("=== Basic SpMV ===")

    A_dense = np.array([
        [0, 0, 1, 0],
        [2, 0, 0, 0],
        [0, 3, 0, 4],
        [0, 0, 0, 0],
    ], dtype=np.float32)

    A = csr_matrix(A_dense)
    x = np.array([1, 2, 3, 4], dtype=np.float32)

    y_scipy = (A @ x).astype(np.float32)
    y_spardial = spmv(A, x)

    print(f"scipy: {y_scipy}")
    print(f"spardial: {y_spardial}")
    # CHECK: scipy: [ 3.  2. 22.  0.]
    # CHECK: spardial: [ 3.  2. 22.  0.]

    assert np.allclose(y_scipy, y_spardial), "Results do not match!"
    print("PASSED")


def test_spmv_identity():
    """Identity matrix test"""
    # CHECK-LABEL: === Identity SpMV ===
    print("=== Identity SpMV ===")

    n = 4
    A = csr_matrix(np.eye(n, dtype=np.float32))
    x = np.array([1, 2, 3, 4], dtype=np.float32)

    y = spmv(A, x)
    print(f"result: {y}")
    # CHECK: result: [1. 2. 3. 4.]

    assert np.allclose(x, y), "Identity SpMV failed!"
    print("PASSED")


def test_spmv_empty():
    """Empty matrix (nnz=0) test"""
    # CHECK-LABEL: === Empty SpMV ===
    print("=== Empty SpMV ===")

    A = csr_matrix((4, 4), dtype=np.float32)
    x = np.array([1, 2, 3, 4], dtype=np.float32)

    y = spmv(A, x)
    print(f"result: {y}")
    # CHECK: result: [0. 0. 0. 0.]

    assert np.allclose(y, np.zeros(4)), "Empty SpMV failed!"
    print("PASSED")


def test_spmv_column_vector():
    """Column vector input (n,1) test"""
    # CHECK-LABEL: === Column Vector SpMV ===
    print("=== Column Vector SpMV ===")

    A = csr_matrix(np.eye(4, dtype=np.float32))
    x = np.array([[1], [2], [3], [4]], dtype=np.float32)  # (4,1)

    y = spmv(A, x)
    print(f"result shape: {y.shape}")
    # CHECK: result shape: (4,)

    assert y.shape == (4,), "Output should be 1D"
    print("PASSED")


if __name__ == "__main__":
    test_spmv_basic()
    test_spmv_identity()
    test_spmv_empty()
    test_spmv_column_vector()
