#!/usr/bin/env python3
"""Test Torch Dialect to Linalg lowering"""

import torch
import sys
import os

# Set PYTHONPATH to the build directory (if running from the build directory)
build_dir = os.path.join(os.path.dirname(__file__), 'build')
python_packages = os.path.join(build_dir, 'python_packages', 'spardial')
if os.path.exists(python_packages):
    sys.path.insert(0, python_packages)

from spardial.importer import import_pytorch_model, lower_to_linalg, print_mlir
from spardial.models import AddNet, MulNet


def test_add_linalg():
    print("\n" + "="*80)
    print("Test: AddNet -> Linalg")
    print("="*80)

    model = AddNet()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    # Step 1: PyTorch -> Torch Dialect
    mlir_module = import_pytorch_model(model, x, y)
    print("\nStep 1: Torch Dialect IR")
    print_mlir(mlir_module)

    # Step 2: Torch Dialect -> Linalg
    linalg_module = lower_to_linalg(mlir_module)
    print("\nStep 2: Linalg-on-Tensors IR")
    print_mlir(linalg_module)


def test_mul_linalg():
    print("\n" + "="*80)
    print("Test: MulNet -> Linalg")
    print("="*80)

    model = MulNet()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    mlir_module = import_pytorch_model(model, x, y)
    linalg_module = lower_to_linalg(mlir_module)
    print_mlir(linalg_module)


if __name__ == "__main__":
    test_add_linalg()
    test_mul_linalg()

    print("\n✅ All Linalg lowering tests passed!")
