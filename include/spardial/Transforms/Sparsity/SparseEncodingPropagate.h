//===- SparseEncodingPropagate.h - Sparse encoding propagation -*- C++ -*-===//
//
// Part of the SparDial Project
//
//===----------------------------------------------------------------------===//

#ifndef SPARDIAL_TRANSFORMS_SPARSITY_SPARSE_ENCODING_PROPAGATE_H
#define SPARDIAL_TRANSFORMS_SPARSITY_SPARSE_ENCODING_PROPAGATE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace spardial {

/// Creates a pass that propagates sparse tensor encodings through the IR.
std::unique_ptr<OperationPass<func::FuncOp>> createSparseEncodingPropagationPass();

} // namespace spardial
} // namespace mlir

#endif // SPARDIAL_TRANSFORMS_SPARSITY_SPARSE_ENCODING_PROPAGATE_H
