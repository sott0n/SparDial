# How Sparse Encoding Works in SparDial

## Answer to "Where does sparse encoding happen?"

**Sparse encoding is automatically added by torch-mlir's FxImporter when it detects PyTorch sparse tensors.**

## The Mechanism

### 1. PyTorch Sparse Tensor Detection

When FxImporter imports an `ExportedProgram`, it examines the actual tensor values passed as arguments. The key code is in `fx_importer.py`:

```python
# In torch_mlir/extras/fx_importer.py, line 1048-1060
if val is not None and val.layout in [
    torch.sparse_coo,
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
]:
    # This is a sparse tensor.
    encoding = sparsity_encoding(val)  # Generate encoding from PyTorch layout
    return IrType.parse(
        f"!{stem}<[{shape_asm}],{str(mlir_dtype)},{encoding}>",
        context=self._c,
    )
```

### 2. Sparse Encoding Generation

The `sparsity_encoding()` function (line 384-440) converts PyTorch sparse layout to MLIR sparse tensor encoding:

```python
def sparsity_encoding(t: torch.Tensor) -> str:
    # ...
    if t.layout is torch.sparse_csr:
        assert sparse_dim == 2
        lvls = f"d{batch_dim}:dense,d{batch_dim+1}:compressed"
        idx_dtype = t.col_indices().dtype
    # ...
    return f"#sparse_tensor.encoding<{{map=({dims})->({lvls}),posWidth={posw},crdWidth={crdw}}}>"
```

For CSR (Compressed Sparse Row), this generates:
```
#sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 64, crdWidth = 64 }>
```

## Complete Pipeline Example

### Input: PyTorch CSR Tensor
```python
A = torch.tensor([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0],
    [0.0, 0.0, 0.0, 0.0],
    [3.0, 0.0, 0.0, 0.0],
])
S = A.to_sparse_csr()  # CSR layout
```

### Step 1: Export and Import (Torch Dialect)
```python
prog = torch.export.export(model, S, Y)  # Pass sparse tensor directly!
fx_importer.import_frozen_program(prog)
```

**Result:** Torch Dialect with sparse encoding
```mlir
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func @main(%arg0: !torch.vtensor<[4,4],f32,#sparse>, ...) {
  ...
}
```

### Step 2: Lower to Linalg
```python
linalg_module = lower_to_linalg(mlir_module)
```

**Result:** Linalg IR with sparse encoding preserved
```mlir
#sparse = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
func.func @main(%arg0: tensor<4x4xf32, #sparse>, ...) {
  %1 = linalg.generic {...} ins(%arg0 : tensor<4x4xf32, #sparse>) {
    ...
  }
}
```

### Step 3: Sparsification and Bufferization
```python
bufferized_module = sparsify_and_bufferize(linalg_module)
```

**Result:** Optimized sparse code with SCF loops
```mlir
func.func @main(%arg0: memref<?xi64>,  // crow_indices
                %arg1: memref<?xi64>,  // col_indices
                %arg2: memref<?xf32>,  // values
                ...) {
  scf.while (...) {  // Sparse iteration
    ...
  }
}
```

## Key Insights

### 1. No Manual Annotation Required
- FxImporter automatically detects `tensor.layout` property
- No need to manually add `#sparse_tensor.encoding` attributes
- No regex or string manipulation needed

### 2. Pass Sparse Tensors Directly
```python
# ✓ CORRECT: Pass sparse tensor directly
S = A.to_sparse_csr()
mlir_module = import_pytorch_model(model, S, Y)

# ✗ WRONG: Converting to dense loses sparsity information
S_dense = S.to_dense()
mlir_module = import_pytorch_model(model, S_dense, Y)  # Will be treated as dense!
```

### 3. Skip Decompositions for Sparse Tensors
PyTorch's decomposition passes don't handle sparse tensors well (stride() errors). SparDial automatically skips decompositions when sparse tensors are detected:

```python
# In spardial/pipeline.py
has_sparse = any(
    hasattr(arg, 'layout') and arg.layout in [
        torch.sparse_coo, torch.sparse_csr, ...
    ]
    for arg in example_args
)

if decomposition_table and not has_sparse:
    prog = prog.run_decompositions(decomposition_table)
elif has_sparse:
    print("Skipping decompositions for sparse tensors...")
```

## Summary

**Where does sparse encoding happen?**
→ **In torch-mlir's FxImporter, in the `sparsity_encoding()` function**

**When does it happen?**
→ **During `import_frozen_program()`, when FxImporter examines input tensor layouts**

**What triggers it?**
→ **PyTorch tensor's `.layout` property (e.g., `torch.sparse_csr`)**

**Do we need manual annotation?**
→ **No! It's completely automatic when you pass sparse tensors directly to the import pipeline**
