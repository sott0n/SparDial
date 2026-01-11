//===- Passes.h - SparDial transform passes --------------------*- C++ -*-===//
//
// Part of the SparDial Project
//
//===----------------------------------------------------------------------===//

#ifndef SPARDIAL_TRANSFORMS_PASSES_H
#define SPARDIAL_TRANSFORMS_PASSES_H

namespace mlir {
namespace spardial {

/// Registers all SparDial transform passes.
void registerTransformPasses();

} // namespace spardial
} // namespace mlir

#endif // SPARDIAL_TRANSFORMS_PASSES_H
