//===- SparseEncodingPropagate.cpp - Sparse encoding propagation ---------===//
//
// Part of the SparDial Project
//
//===----------------------------------------------------------------------===//

#include "spardial/Transforms/Sparsity/SparseEncodingPropagate.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sparse-encoding-propagation"

namespace mlir {
#define GEN_PASS_DEF_SPARSEENCODINGPROPAGATION
#include "spardial/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Pass implementation for sparse encoding propagation.
struct SparseEncodingPropagation
    : public impl::SparseEncodingPropagationBase<SparseEncodingPropagation> {

  SparseEncodingPropagation() = default;
  SparseEncodingPropagation(const SparseEncodingPropagation &pass) = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Running SparseEncodingPropagation on function: " << funcOp.getName()
                            << "\n");

    // Collect sparse tensor encoding information from function arguments
    llvm::DenseMap<Value, sparse_tensor::SparseTensorEncodingAttr> sparseEncodings;

    for (auto arg : funcOp.getArguments()) {
      if (auto tensorType = dyn_cast<RankedTensorType>(arg.getType())) {
        if (auto encoding = dyn_cast_or_null<sparse_tensor::SparseTensorEncodingAttr>(
                tensorType.getEncoding())) {
          sparseEncodings[arg] = encoding;
          LLVM_DEBUG(llvm::dbgs() << "  Found sparse argument: " << arg << "\n");
        }
      }
    }

    // Propagate sparse encodings through operations
    funcOp.walk([&](linalg::GenericOp genericOp) {
      LLVM_DEBUG(llvm::dbgs() << "  Processing linalg.generic operation\n");

      // Check if any inputs have sparse encodings
      sparse_tensor::SparseTensorEncodingAttr inputEncoding = nullptr;
      for (auto input : genericOp.getDpsInputs()) {
        if (sparseEncodings.count(input)) {
          inputEncoding = sparseEncodings[input];
          LLVM_DEBUG(llvm::dbgs() << "    Input has sparse encoding\n");
          break;
        }
      }

      // If we found a sparse input, propagate to outputs
      if (inputEncoding) {
        for (auto output : genericOp.getDpsInits()) {
          if (auto tensorType = dyn_cast<RankedTensorType>(output.getType())) {
            // Only propagate if output doesn't already have an encoding
            if (!tensorType.getEncoding()) {
              LLVM_DEBUG(llvm::dbgs() << "    Propagating encoding to output: " << output << "\n");

              // Record this value's sparse encoding for further propagation
              sparseEncodings[output] = inputEncoding;

              // Note: Actually modifying the IR to change types would require
              // more complex rewriting logic. This pass currently only analyzes
              // and marks opportunities for sparsification.
              LLVM_DEBUG({
                auto newType = RankedTensorType::get(tensorType.getShape(),
                                                     tensorType.getElementType(), inputEncoding);
                llvm::dbgs() << "    Marked output for sparse encoding: " << newType << "\n";
              });
            }
          }
        }
      }
    });

    // Additional propagation through tensor operations
    funcOp.walk([&](tensor::EmptyOp emptyOp) {
      // Check if this empty tensor is used by operations with sparse inputs
      for (auto user : emptyOp.getResult().getUsers()) {
        if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
          // Check if this operation has sparse inputs
          for (auto input : genericOp.getDpsInputs()) {
            if (sparseEncodings.count(input)) {
              LLVM_DEBUG(llvm::dbgs() << "  Marking tensor.empty for sparsification\n");
              sparseEncodings[emptyOp.getResult()] = sparseEncodings[input];
              break;
            }
          }
        }
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "  Total values marked as sparse: " << sparseEncodings.size()
                            << "\n");
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::spardial::createSparseEncodingPropagationPass() {
  return std::make_unique<SparseEncodingPropagation>();
}
