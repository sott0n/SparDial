# RUN: %PYTHON %s | FileCheck %s

"""
Test SpMM with NumPy/SciPy backend.
Tests basic SpMM, rectangular matrices, and float64.
"""

from scipy.sparse import csr_matrix
import numpy as np
from spardial import spardial_jit


@spardial_jit
def spmm(A, B):
    return A @ B


def test_spmm_basic():
    """Basic SpMM test"""
    # CHECK-LABEL: === Basic SpMM ===
    print("=== Basic SpMM ===")

    A_dense = np.array([
        [1, 0, 2],
        [0, 3, 0],
    ], dtype=np.float32)

    B = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ], dtype=np.float32)

    A = csr_matrix(A_dense)
    C_scipy = (A @ B).astype(np.float32)
    C_spardial = spmm(A, B)

    print(f"scipy: {C_scipy}")
    print(f"spardial: {C_spardial}")
    # CHECK: scipy: {{\[\[11\. 14\.\]}}
    # CHECK: spardial: {{\[\[11\. 14\.\]}}

    assert np.allclose(C_scipy, C_spardial), "Results do not match!"
    print("PASSED")


def test_spmm_rectangular():
    """Rectangular SpMM test"""
    # CHECK-LABEL: === Rectangular SpMM ===
    print("=== Rectangular SpMM ===")

    A = csr_matrix(np.array([
        [0, 1, 0, 2],
        [3, 0, 4, 0],
    ], dtype=np.float32))
    B = np.array([
        [1, 0, 2],
        [0, 3, 0],
        [4, 0, 5],
        [0, 6, 0],
    ], dtype=np.float32)

    C = spmm(A, B)
    print(f"result shape: {C.shape}")
    # CHECK: result shape: (2, 3)

    assert C.shape == (2, 3), "Output shape mismatch!"
    assert np.allclose(C, (A @ B)), "Rectangular SpMM failed!"
    print("PASSED")


def test_spmm_float64():
    """Float64 SpMM test"""
    # CHECK-LABEL: === Float64 SpMM ===
    print("=== Float64 SpMM ===")

    A = csr_matrix(np.array([
        [1.5, 0.0],
        [0.0, 2.0],
    ], dtype=np.float64))
    B = np.array([
        [2.0, 3.0],
        [4.0, 5.0],
    ], dtype=np.float64)

    C_scipy = (A @ B).astype(np.float64)
    C_spardial = spmm(A, B)

    print(f"scipy: {C_scipy}")
    print(f"spardial: {C_spardial}")
    # CHECK: scipy: {{\[\[[ ]*3\. 4\.5\]}}
    # CHECK: spardial: {{\[\[[ ]*3\. 4\.5\]}}

    assert np.allclose(C_scipy, C_spardial), "Float64 results do not match!"
    print("PASSED")


if __name__ == "__main__":
    test_spmm_basic()
    test_spmm_rectangular()
    test_spmm_float64()
