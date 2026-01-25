//===- SparDialModule.cpp - SparDial Python extension --------------------===//
//
// Part of the SparDial Project
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "spardial-c/Registration.h"

PYBIND11_MODULE(_spardial, m) {
  // Register all SparDial passes
  spardialRegisterAllPasses();

  m.doc() = "SparDial main python extension";

  m.def(
      "register_passes", []() { spardialRegisterAllPasses(); }, "Register all SparDial passes");
}
