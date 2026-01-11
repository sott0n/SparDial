# SparDial C API library
add_mlir_public_c_api_library(SparDialCAPI
    Registration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/spardial-c

    ENABLE_AGGREGATION

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    SparDialTransformPasses
)
