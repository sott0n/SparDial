//===- Registration.cpp - SparDial C API Registration --------------------===//
//
// Part of the SparDial Project
//
//===----------------------------------------------------------------------===//

#include "spardial-c/Registration.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "spardial/Transforms/Passes.h"

MLIR_CAPI_EXPORTED void spardialRegisterAllPasses() { mlir::spardial::registerTransformPasses(); }
