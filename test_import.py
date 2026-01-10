#!/usr/bin/env python3
"""Test PyTorch to MLIR import"""

import torch
import sys
import os

# Set PYTHONPATH to the build directory (if running from the build directory)
build_dir = os.path.join(os.path.dirname(__file__), 'build')
python_packages = os.path.join(build_dir, 'python_packages', 'spardial')
if os.path.exists(python_packages):
    sys.path.insert(0, python_packages)

from spardial.importer import import_pytorch_model, print_mlir
from spardial.models import AddNet, MulNet, SimpleLinear


def test_add():
    print("\n" + "="*80)
    print("Test 1: AddNet")
    print("="*80)

    model = AddNet()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    mlir_module = import_pytorch_model(model, x, y)
    print_mlir(mlir_module)


def test_mul():
    print("\n" + "="*80)
    print("Test 2: MulNet")
    print("="*80)

    model = MulNet()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    mlir_module = import_pytorch_model(model, x, y)
    print_mlir(mlir_module)


def test_linear():
    print("\n" + "="*80)
    print("Test 3: SimpleLinear")
    print("="*80)

    model = SimpleLinear(in_features=4, out_features=2)
    x = torch.randn(1, 4)

    mlir_module = import_pytorch_model(model, x)
    print_mlir(mlir_module)


if __name__ == "__main__":
    test_add()
    test_mul()
    test_linear()

    print("\n✅ All import tests passed!")
