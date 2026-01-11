# SparDial Sparsity Propagation Pass
add_mlir_library(SparDialSparsityPropagation
    SparseEncodingPropagate.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/spardial/Transforms/Sparsity

    DEPENDS
    SparDialTransformsPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRSparseTensorDialect
    MLIRTensorDialect
    MLIRSupport
)
