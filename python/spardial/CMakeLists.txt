set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(SPARDIAL_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/spardial")

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=spardial.")

include(MLIRDialectPythonEnv)
include(AddMLIRPython)

include()
