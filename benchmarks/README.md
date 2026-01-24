# SparDial Benchmarks

Performance comparison between PyTorch CPU execution and SparDial JIT CPU execution.

## Quick Start

```shell
# Build the project first
cd build && ninja SparDialPythonModules

# Run quick benchmark (sanity check)
ninja spardial-benchmark-quick

# Run medium benchmark (recommended)
ninja spardial-benchmark-medium

# Run full benchmark (comprehensive)
ninja spardial-benchmark
```

## Available Targets

| Target | Description |
|--------|-------------|
| `spardial-benchmark` | Full benchmark: 100x100 to 5000x5000, sparsity 90-99% |
| `spardial-benchmark-medium` | Medium benchmark: 500x500 to 2000x2000, sparsity 90-99% |
| `spardial-benchmark-quick` | Quick sanity check: 50x50, 3 iterations |
| `spardial-benchmark-json` | Full benchmark with JSON output |
| `spardial-benchmark-csv` | Full benchmark with CSV output |

## CLI Usage

You can also run benchmarks directly via Python:

```shell
# Set up PYTHONPATH
export PYTHONPATH=/path/to/build/tools/spardial/python_packages/spardial:/path/to/SparDial

# Run with custom parameters
python -m benchmarks --benchmarks spmv,spmm,add --sizes 1000x1000,2000x2000 --sparsities 0.9,0.99 --formats csr

# Quick mode
python -m benchmarks --quick --progress

# Output formats
python -m benchmarks --output json > results.json
python -m benchmarks --output csv > results.csv
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--benchmarks` | Comma-separated list (spmv,spmm,add) | all |
| `--sizes` | Matrix sizes (e.g., 100x100,500x500) | 100x100 |
| `--sparsities` | Sparsity levels (e.g., 0.9,0.95,0.99) | 0.9 |
| `--formats` | Sparse formats (csr,csc,coo,dense) | csr |
| `--warmup` | Warmup iterations | 3 |
| `--iterations` | Benchmark iterations | 10 |
| `--output` | Output format (human,json,csv) | human |
| `--quick` | Quick mode (small sizes, fewer iterations) | false |
| `--progress` | Print progress for each benchmark case | false |

## Benchmark Kernels

- **SpMV** (Sparse Matrix-Vector Multiplication): `MVNet` - `torch.mv(sparse, dense)`
- **SpMM** (Sparse Matrix-Matrix Multiplication): `MMNet` - `torch.mm(sparse, sparse)`
- **Add** (Sparse Addition): `AddNet` - `torch.add(sparse, sparse)`

---

## Benchmark Results

### Environment

- **Date**: 2026-01-24
- **Commit**: bd94c05
- **Platform**: Linux x86_64
- **Python**: 3.11
- **PyTorch**: 2.x (with sparse tensor support)

### SpMV (Sparse Matrix-Vector Multiplication)

| Size | Sparsity | PyTorch (ms) | SparDial (ms) | Speedup |
|------|----------|--------------|---------------|---------|
| 500x500 | 90% | 0.044 | 0.260 | 0.17x |
| 500x500 | 95% | 0.032 | 0.256 | 0.12x |
| 500x500 | 99% | 0.023 | 0.253 | 0.09x |
| 1000x1000 | 90% | 0.059 | 0.311 | 0.19x |
| 1000x1000 | 95% | 0.055 | 0.278 | 0.20x |
| 1000x1000 | 99% | 0.031 | 0.256 | 0.12x |
| 2000x2000 | 90% | 0.127 | 0.523 | 0.24x |
| 2000x2000 | 95% | 0.088 | 0.376 | 0.23x |
| 2000x2000 | 99% | 0.050 | 0.275 | 0.18x |

### SpMM (Sparse Matrix-Matrix Multiplication)

| Size | Sparsity | PyTorch (ms) | SparDial (ms) | Speedup |
|------|----------|--------------|---------------|---------|
| 500x500 | 90% | 0.88 | 3.44 | 0.26x |
| 500x500 | 95% | 0.60 | 3.14 | 0.19x |
| 500x500 | 99% | 0.12 | 1.75 | 0.07x |
| 1000x1000 | 90% | 4.04 | 15.99 | 0.25x |
| 1000x1000 | 95% | 3.35 | 11.44 | 0.29x |
| 1000x1000 | 99% | 0.36 | 5.46 | 0.07x |
| 2000x2000 | 90% | 17.01 | 87.26 | 0.20x |
| 2000x2000 | 95% | 10.88 | 50.52 | 0.22x |
| 2000x2000 | 99% | 1.60 | 24.52 | 0.07x |

### Add (Sparse Addition)

| Size | Sparsity | PyTorch (ms) | SparDial (ms) | Speedup |
|------|----------|--------------|---------------|---------|
| 500x500 | 90% | 0.25 | 0.87 | 0.29x |
| 500x500 | 95% | 0.15 | 0.66 | 0.22x |
| 500x500 | 99% | 0.07 | 0.49 | 0.14x |
| 1000x1000 | 90% | 0.89 | 2.29 | 0.39x |
| 1000x1000 | 95% | 0.50 | 1.28 | 0.39x |
| 1000x1000 | 99% | 0.15 | 0.62 | 0.24x |
| 2000x2000 | 90% | 4.71 | 10.42 | 0.45x |
| 2000x2000 | 95% | 1.78 | 4.28 | 0.42x |
| 2000x2000 | 99% | 0.45 | 1.13 | 0.40x |

### Analysis

**Current Status**: SparDial is currently slower than PyTorch for these benchmarks.

**Observations**:

1. **Size scaling**: Speedup improves with larger matrices
   - Add @ 2000x2000: 0.45x (best observed)
   - SpMV @ 2000x2000: 0.24x

2. **Sparsity impact**: Higher sparsity (99%) tends to reduce relative performance
   - PyTorch's sparse kernels efficiently skip zero elements
   - SparDial's generated code has fixed overhead

3. **Operation complexity**: Simpler operations (Add) show better relative performance than complex ones (SpMM)

**Why SparDial is slower**:

- PyTorch uses hand-tuned, vectorized (SIMD) sparse kernels
- SparDial generates generic code without specialized optimizations
- JIT compilation overhead (not included in runtime, but ~40-400ms per kernel)
- Limited parallelization in current implementation

**Potential improvements**:

- Enable vectorization in MLIR sparsification
- Add parallelization (OpenMP/threading)
- Optimize for specific sparse patterns
- Improve code generation for CSR format
