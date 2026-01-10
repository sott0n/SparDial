#!/usr/bin/env python3
"""Test PyTorch to MLIR compilation pipeline"""

import torch
import sys
import os

# Set PYTHONPATH to the build directory
build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
python_packages = os.path.join(build_dir, 'tools', 'spardial', 'python_packages', 'spardial')
if os.path.exists(python_packages):
    sys.path.insert(0, python_packages)

from spardial.importer import import_pytorch_model, lower_to_linalg, print_mlir
from spardial.models import AddNet, MulNet, SimpleLinear


def test_torch_import():
    """Test PyTorch to Torch Dialect import"""
    print("\n" + "="*80)
    print("Test 1: PyTorch -> Torch Dialect (AddNet)")
    print("="*80)

    model = AddNet()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    mlir_module = import_pytorch_model(model, x, y)
    print_mlir(mlir_module)


def test_torch_import_mul():
    """Test PyTorch to Torch Dialect import for MulNet"""
    print("\n" + "="*80)
    print("Test 2: PyTorch -> Torch Dialect (MulNet)")
    print("="*80)

    model = MulNet()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    mlir_module = import_pytorch_model(model, x, y)
    print_mlir(mlir_module)


def test_torch_import_linear():
    """Test PyTorch to Torch Dialect import for SimpleLinear"""
    print("\n" + "="*80)
    print("Test 3: PyTorch -> Torch Dialect (SimpleLinear)")
    print("="*80)

    model = SimpleLinear(in_features=4, out_features=2)
    x = torch.randn(1, 4)

    mlir_module = import_pytorch_model(model, x)
    print_mlir(mlir_module)


def test_linalg_lowering_add():
    """Test full pipeline: PyTorch -> Torch Dialect -> Linalg (AddNet)"""
    print("\n" + "="*80)
    print("Test 4: Full Pipeline - AddNet -> Linalg")
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


def test_linalg_lowering_mul():
    """Test full pipeline: PyTorch -> Torch Dialect -> Linalg (MulNet)"""
    print("\n" + "="*80)
    print("Test 5: Full Pipeline - MulNet -> Linalg")
    print("="*80)

    model = MulNet()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    mlir_module = import_pytorch_model(model, x, y)
    linalg_module = lower_to_linalg(mlir_module)

    print("\nLinalg-on-Tensors IR:")
    print_mlir(linalg_module)


if __name__ == "__main__":
    # Test PyTorch to Torch Dialect import
    test_torch_import()
    test_torch_import_mul()
    test_torch_import_linear()

    # Test full pipeline to Linalg
    test_linalg_lowering_add()
    test_linalg_lowering_mul()

    print("\n" + "="*80)
    print("✅ All pipeline tests passed!")
    print("="*80)
