---
name: ask-mlir-docs
description: References MLIR documentation (local and online) for SparDial development. Use when asking about MLIR concepts, Sparse Tensor dialect, Linalg, bufferization, or pass infrastructure.
allowed-tools: WebFetch, Read, Grep, Glob
---

# MLIR Documentation Reference

## Local Documentation

Base path: `$HOME/SparDial/externals/torch-mlir/externals/llvm-project/`

### Core Docs (`mlir/docs/`)

| File | Topic |
|------|-------|
| `LangRef.md` | MLIR language reference |
| `Bufferization.md` | Bufferization infrastructure |
| `Passes.md` | Pass management |
| `DialectConversion.md` | Dialect conversion framework |
| `Interfaces.md` | Interfaces and traits |

### Dialect Docs (`mlir/docs/Dialects/`)

| File | Dialect |
|------|---------|
| `Linalg/_index.md` | Linalg |
| `MemRef.md` | MemRef |
| `Func.md` | Func |
| `Vector.md` | Vector |
| `LLVM.md` | LLVM |

### SparseTensor (`mlir/include/mlir/Dialect/SparseTensor/IR/`)

| File | Content |
|------|---------|
| `SparseTensorOps.td` | Operations |
| `SparseTensorAttrDefs.td` | Encoding attributes |
| `SparseTensorTypes.td` | Types |

## Online Documentation

| Topic | URL |
|-------|-----|
| Sparse Tensor Ops | https://mlir.llvm.org/docs/Dialects/SparseTensorOps/ |
| Linalg | https://mlir.llvm.org/docs/Dialects/Linalg/ |
| Bufferization | https://mlir.llvm.org/docs/Bufferization/ |
| Pass Management | https://mlir.llvm.org/docs/PassManagement/ |
| MLIR Pass list | https://mlir.llvm.org/docs/Passes/ |
| Dialect Lists | https://mlir.llvm.org/docs/Dialects/ |
| Tracing and Debugging | https://mlir.llvm.org/docs/ActionTracing/ |
| Data Layout Modeling | https://mlir.llvm.org/docs/DataLayout/ |
| Defining Dialects | https://mlir.llvm.org/docs/DefiningDialects/ |
| Dialect Conversion | https://mlir.llvm.org/docs/DialectConversion/ |
| Interface | https://mlir.llvm.org/docs/Interfaces/ |
| MLIR Bytecode Format | https://mlir.llvm.org/docs/BytecodeFormat/ |
| MLIR C API | https://mlir.llvm.org/docs/CAPI/ |
| Operation Canonicalization | https://mlir.llvm.org/docs/Canonicalization/ |
| Ownership-based Buffer Deallocation | https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/ |
| Symbols and Symbol Table | https://mlir.llvm.org/docs/SymbolsAndSymbolTables/ |

## Usage

- Prefer local files (faster, no network)
- Use online docs for SparseTensor (more comprehensive than TableGen)
- For SparDial-specific info, see `CLAUDE.md` in project root
