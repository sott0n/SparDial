"""Test PyTorch to MLIR compilation pipeline"""

import torch
import pytest

from spardial.importer import import_pytorch_model, lower_to_linalg
from spardial.models import AddNet, MulNet, SimpleLinear


class TestTorchDialectImport:
    """Test PyTorch to Torch Dialect IR conversion"""

    def test_addnet_import(self):
        """Test AddNet model import to Torch Dialect"""
        model = AddNet()
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        mlir_module = import_pytorch_model(model, x, y)

        # Verify module is created
        assert mlir_module is not None
        module_str = str(mlir_module)

        # Verify Torch Dialect operations are present
        assert 'torch.aten.add.Tensor' in module_str
        assert '!torch.vtensor<[2,3],f32>' in module_str
        assert 'func.func @main' in module_str

    def test_mulnet_import(self):
        """Test MulNet model import to Torch Dialect"""
        model = MulNet()
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        mlir_module = import_pytorch_model(model, x, y)

        # Verify module is created
        assert mlir_module is not None
        module_str = str(mlir_module)

        # Verify Torch Dialect operations are present
        assert 'torch.aten.mul.Tensor' in module_str
        assert '!torch.vtensor<[2,3],f32>' in module_str

    def test_linear_import(self):
        """Test SimpleLinear model import to Torch Dialect"""
        model = SimpleLinear(in_features=4, out_features=2)
        x = torch.randn(1, 4)

        mlir_module = import_pytorch_model(model, x)

        # Verify module is created
        assert mlir_module is not None
        module_str = str(mlir_module)

        # Verify Torch Dialect operations are present
        assert 'torch.aten.linear' in module_str
        assert '!torch.vtensor<[1,4],f32>' in module_str
        assert '!torch.vtensor<[1,2],f32>' in module_str


class TestLinalgLowering:
    """Test Torch Dialect to Linalg-on-Tensors IR conversion"""

    def test_addnet_lowering(self):
        """Test AddNet lowering to Linalg IR"""
        model = AddNet()
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # Step 1: PyTorch -> Torch Dialect
        mlir_module = import_pytorch_model(model, x, y)
        assert mlir_module is not None

        # Step 2: Torch Dialect -> Linalg
        linalg_module = lower_to_linalg(mlir_module)
        assert linalg_module is not None

        module_str = str(linalg_module)

        # Verify Linalg operations are present
        assert 'linalg.generic' in module_str
        assert 'arith.addf' in module_str
        assert 'tensor<2x3xf32>' in module_str
        assert 'affine_map' in module_str

        # Verify Torch Dialect is lowered away
        assert 'torch.aten' not in module_str
        assert '!torch.vtensor' not in module_str

    def test_mulnet_lowering(self):
        """Test MulNet lowering to Linalg IR"""
        model = MulNet()
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # Step 1: PyTorch -> Torch Dialect
        mlir_module = import_pytorch_model(model, x, y)
        assert mlir_module is not None

        # Step 2: Torch Dialect -> Linalg
        linalg_module = lower_to_linalg(mlir_module)
        assert linalg_module is not None

        module_str = str(linalg_module)

        # Verify Linalg operations are present
        assert 'linalg.generic' in module_str
        assert 'arith.mulf' in module_str
        assert 'tensor<2x3xf32>' in module_str

        # Verify Torch Dialect is lowered away
        assert 'torch.aten' not in module_str


class TestEndToEndPipeline:
    """Test complete PyTorch to Linalg pipeline"""

    @pytest.mark.parametrize("model_class,input_shape", [
        (AddNet, [(2, 3), (2, 3)]),
        (MulNet, [(2, 3), (2, 3)]),
    ])
    def test_full_pipeline(self, model_class, input_shape):
        """Test full pipeline for various models"""
        model = model_class()
        inputs = [torch.randn(*shape) for shape in input_shape]

        # Import to Torch Dialect
        mlir_module = import_pytorch_model(model, *inputs)
        assert mlir_module is not None
        assert 'torch.aten' in str(mlir_module)

        # Lower to Linalg
        linalg_module = lower_to_linalg(mlir_module)
        assert linalg_module is not None

        linalg_str = str(linalg_module)
        assert 'linalg.generic' in linalg_str
        assert 'torch.aten' not in linalg_str
