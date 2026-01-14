set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(SPARDIAL_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/spardial")

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=spardial.")

include(MLIRDetectPythonEnv)
include(AddMLIRPython)
mlir_configure_python_dev_packages()

declare_mlir_python_sources(SparDialPythonSources)

declare_mlir_python_sources(SparDialPythonSources.Core
    ROOT_DIR "${SPARDIAL_PYTHON_ROOT_DIR}"
    ADD_TO_PARENT SparDialPythonSources
    SOURCES
        __init__.py
        pipeline.py
        models.py
        backend.py
        models/__init__.py
        models/kernels.py
)

# Declare parent for SparDial extensions
declare_mlir_python_sources(SparDialPythonExtensions)

# SparDial Python extension for passes
declare_mlir_python_extension(SparDialPythonExtensions.Main
    MODULE_NAME _spardial
    ADD_TO_PARENT SparDialPythonExtensions
    SOURCES
        SparDialModule.cpp
    EMBED_CAPI_LINK_LIBS
        SparDialCAPI
    PRIVATE_LINK_LIBS
        LLVMSupport
)

set(_source_components
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything

    TorchMLIRPythonSources
    TorchMLIRPythonExtensions

    SparDialPythonSources
    SparDialPythonExtensions
)

add_mlir_python_common_capi_library(SparDialAggregateCAPI
    INSTALL_COMPONENT SparDialPythonModules
    INSTALL_DESTINATION python_packages/spardial/spardial/_mlir_libs
    OUTPUT_DIRECTORY "${SPARDIAL_BINARY_DIR}/python_packages/spardial/spardial/_mlir_libs"
    RELATIVE_INSTALL_ROOT ".."
    DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(SparDialPythonModules
    ROOT_PREFIX "${SPARDIAL_BINARY_DIR}/python_packages/spardial/spardial"
    INSTALL_PREFIX "python_packages/spardial/spardial"
    DECLARED_SOURCES ${_source_components}
    COMMON_CAPI_LINK_LIBS SparDialAggregateCAPI
)
