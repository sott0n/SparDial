# Add subdirectories
add_subdirectory(Sparsity)

# Aggregate all transform libraries
set(linked_libs
    SparDialSparsityPropagation
)

# Main transform passes library
add_mlir_library(SparDialTransformPasses
    Passes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/spardial/Transforms

    DEPENDS
    SparDialTransformsPassIncGen

    LINK_LIBS PUBLIC
    ${linked_libs}
)
