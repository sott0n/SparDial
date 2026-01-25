## NumPy Tracing (SpMV MVP) Design & Plan

### Goal
Enable `@spardial_jit` to trace a NumPy/SciPy function and JIT-compile **SpMV** via MLIR, without relying on function names.

### Minimal Scope (Phase 1)
- Operation: `A @ x` only (no other NumPy ops).
- Inputs:
  - `A`: `scipy.sparse.csr_matrix` **required**.
  - `x`: `np.ndarray` 1D vector.
- Types: `float32` / `float64` only.
- Shapes: 2D @ 1D only.
- Any deviation → explicit error.

### Design

#### 1) Tracing Entry (`@spardial_jit`)
- Decorator wraps a Python function.
- At call time, inputs are converted to **Proxy** objects.
- Function is executed once with proxies to record a tiny IR.
- Only the `@` operator is supported.

#### 2) Proxy Objects
- `NumpyProxy` (for `np.ndarray`):
  - Implements `__matmul__`.
  - Returns `SpmvOp` node.
- `CsrProxy` (for `csr_matrix`):
  - Implements `__matmul__`.
  - Returns `SpmvOp` node.
- Any other op: raise `NotImplementedError`.

#### 3) Minimal IR
- Single node: `SpmvOp(A, x)`
- Contains:
  - dtype (float32/float64)
  - shapes (A: 2D, x: 1D)
  - CSR metadata check (index dtype: int32/int64)
- No graph optimizations for MVP.

#### 4) Lowering to MLIR
- Reuse existing components:
  - `KernelBuilder` for SpMV MLIR generation
  - `SparseEncodingBuilder` for CSR encoding
  - `sparsify_and_bufferize` + `prepare_for_execution`
- Use `jit_runtime` ExecutionEngine path for invocation.

#### 5) Runtime
- Use existing `SparseCompiler.execute_spmv` for actual execution path.
- Return numpy array result.

### Error Handling
- If function contains anything other than `@`, raise `NotImplementedError`.
- If `A` is not CSR → `TypeError`.
- If `x` is not 1D numpy vector → `TypeError`.
- If dtype not float32/float64 → `TypeError`.
- If shapes incompatible → `ValueError`.

### Tests (Phase 1)
- Update `tests/numpy/test_spmv.py` to call a `@spardial_jit`-decorated function.
- Add a negative test for non-CSR input (should raise).
- Add a negative test for non-1D x (should raise).

### Plan (Next Steps)
1. Implement proxy/tracer in `python/spardial/numpy_backend/tracer.py`.
2. Extend `spardial_jit` to:
   - detect NumPy function
   - run tracer once
   - dispatch to SpMV execution path
3. Update tests to exercise traced path.
