# Generate pass declarations from TableGen
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(SparDialTransformsPassIncGen)

# Generate pass documentation
add_mlir_doc(Passes SparDialTransformsPass ./ -gen-pass-doc)
