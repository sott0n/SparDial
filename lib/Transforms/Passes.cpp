//===- Passes.cpp - SparDial transform passes ----------------------------===//
//
// Part of the SparDial Project
//
//===----------------------------------------------------------------------===//

#include "spardial/Transforms/Passes.h"
#include "spardial/Transforms/Sparsity/SparseEncodingPropagate.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "spardial/Transforms/Passes.h.inc"
} // namespace

void mlir::spardial::registerTransformPasses() { ::registerPasses(); }
