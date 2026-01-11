"""Test PyTorch to MLIR compilation pipeline"""

import torch
import pytest

from spardial.importer import import_pytorch_model, lower_to_linalg, sparsify_and_bufferize
from spardial.models import AddNet, MulNet, SimpleLinear
from spardial.passmanager import PassManager


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


class TestCustomPass:
    """Test custom SparDial passes"""

    def test_sparse_encoding_propagation_registration(self):
        """Test that sparse-encoding-propagation pass is registered"""
        # Import _spardial extension to ensure passes are registered
        import spardial._mlir_libs._spardial

        # Create a simple module and verify pass can be parsed
        model = AddNet()
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        mlir_module = import_pytorch_model(model, x, y)
        linalg_module = lower_to_linalg(mlir_module)

        # Try to parse the pass - this will fail if not registered
        with linalg_module.context:
            pm = PassManager.parse(
                "builtin.module(func.func(sparse-encoding-propagation))"
            )
            assert pm is not None

    def test_sparse_encoding_propagation_execution(self):
        """Test sparse-encoding-propagation pass execution"""
        # Import _spardial extension
        import spardial._mlir_libs._spardial

        model = AddNet()
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # Create Linalg IR
        mlir_module = import_pytorch_model(model, x, y)
        linalg_module = lower_to_linalg(mlir_module)

        # Get IR before pass
        ir_before = str(linalg_module)
        assert 'linalg.generic' in ir_before
        assert 'tensor<2x3xf32>' in ir_before

        # Apply the pass
        with linalg_module.context:
            pm = PassManager.parse(
                "builtin.module(func.func(sparse-encoding-propagation))"
            )
            pm.run(linalg_module.operation)

        # Get IR after pass
        ir_after = str(linalg_module)

        # Verify IR structure is maintained
        assert 'linalg.generic' in ir_after
        assert 'tensor<2x3xf32>' in ir_after
        assert 'func.func @main' in ir_after

    @pytest.mark.parametrize("model_class", [AddNet, MulNet])
    def test_sparse_encoding_propagation_various_ops(self, model_class):
        """Test pass on various operations"""
        import spardial._mlir_libs._spardial

        model = model_class()
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        mlir_module = import_pytorch_model(model, x, y)
        linalg_module = lower_to_linalg(mlir_module)

        # Apply pass without errors
        with linalg_module.context:
            pm = PassManager.parse(
                "builtin.module(func.func(sparse-encoding-propagation))"
            )
            pm.run(linalg_module.operation)

        # Verify output is valid MLIR
        ir_after = str(linalg_module)
        assert 'func.func @main' in ir_after
        assert 'linalg.generic' in ir_after


class TestSparsificationBufferization:
    """Test sparsification and bufferization pipeline"""

    def test_sparsify_basic(self):
        """Test basic sparsification pipeline on dense tensors"""
        import spardial._mlir_libs._spardial

        model = AddNet()
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # Create Linalg IR
        mlir_module = import_pytorch_model(model, x, y)
        linalg_module = lower_to_linalg(mlir_module)

        # Verify input is tensor-based
        ir_before = str(linalg_module)
        assert 'tensor<2x3xf32>' in ir_before

        # Apply sparsification and bufferization
        bufferized_module = sparsify_and_bufferize(linalg_module)

        # Verify output is bufferized (memref-based)
        ir_after = str(bufferized_module)
        assert 'memref' in ir_after
        assert 'func.func @main' in ir_after

    def test_sparsify_with_options(self):
        """Test sparsification with custom options"""
        import spardial._mlir_libs._spardial

        model = MulNet()
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        mlir_module = import_pytorch_model(model, x, y)
        linalg_module = lower_to_linalg(mlir_module)

        # Apply with parallelization options
        bufferized_module = sparsify_and_bufferize(
            linalg_module,
            sparse_options="parallelization-strategy=none"
        )

        ir_after = str(bufferized_module)
        assert 'func.func @main' in ir_after
        assert 'memref' in ir_after

    @pytest.mark.parametrize("model_class", [AddNet, MulNet])
    def test_full_pipeline_with_sparsification(self, model_class):
        """Test complete pipeline: PyTorch -> Torch -> Linalg -> Sparse/Bufferized"""
        import spardial._mlir_libs._spardial

        model = model_class()
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        # Step 1: PyTorch -> Torch Dialect
        mlir_module = import_pytorch_model(model, x, y)
        assert mlir_module is not None
        assert 'torch.aten' in str(mlir_module)

        # Step 2: Torch Dialect -> Linalg
        linalg_module = lower_to_linalg(mlir_module)
        assert linalg_module is not None
        assert 'linalg.generic' in str(linalg_module)
        assert 'tensor' in str(linalg_module)

        # Step 3: Linalg -> Sparse & Bufferized
        bufferized_module = sparsify_and_bufferize(linalg_module)
        assert bufferized_module is not None

        ir_final = str(bufferized_module)
        assert 'func.func @main' in ir_final
        assert 'memref' in ir_final
