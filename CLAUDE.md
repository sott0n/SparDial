# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparDial is a Sparse Dialect Compiler based on MLIR (Multi-Level Intermediate Representation).

## Development Status

This repository is in its initial setup phase. As the codebase develops, this file should be updated with:

- Build system commands (CMake, Make, etc.)
- Test execution commands
- Code architecture and dialect structure
- MLIR dialect registration and operation definitions
- Sparse tensor representation patterns
- Integration points with LLVM/MLIR infrastructure

## Expected Architecture

As an MLIR-based compiler project, the codebase will likely include:

- **Dialect definitions**: Custom MLIR operations for sparse computations
- **Transformations**: Passes for optimizing sparse operations
- **Lowering paths**: Converting high-level sparse operations to lower-level representations
- **Runtime support**: Libraries for executing sparse operations

Update this document as the project structure materializes.
