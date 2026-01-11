//===- Registration.h - SparDial C API Registration ------------*- C -*-===//
//
// Part of the SparDial Project
//
//===----------------------------------------------------------------------===//

#ifndef SPARDIAL_C_REGISTRATION_H
#define SPARDIAL_C_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers all SparDial passes with the global pass registry.
MLIR_CAPI_EXPORTED void spardialRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // SPARDIAL_C_REGISTRATION_H
