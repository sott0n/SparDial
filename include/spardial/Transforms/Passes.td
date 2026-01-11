//===- Passes.td - SparDial pass definition file ----------*- tablegen -*-===//
//
// Part of the SparDial Project
//
//===----------------------------------------------------------------------===//

#ifndef SPARDIAL_TRANSFORMS_PASSES
#define SPARDIAL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def SparseEncodingPropagation : Pass<"sparse-encoding-propagation", "func::FuncOp"> {
    let summary = "Propagate sparse tensor encodings through operations";
    let description = [{
        A pass that propagates sparse tensor encoding attributes through the IR.
        This pass analyzes tensor operations and propagates sparsity information
        from inputs to outputs, enabling subsequent optimization passes to generate
        specialized sparse code.

        The pass works on `func.func` operations and traverses the operation graph
        to identify opportunities for sparse tensor encoding propagation. It:

        1. Collects sparse tensor encoding attributes from function arguments
        2. Propagates encodings through linalg.generic operations
        3. Marks tensor.empty operations that feed sparse computations
        4. Analyzes the IR to identify sparsification opportunities

        Note: This pass performs analysis and marking. Actual type transformation
        and sparse code generation would be handled by subsequent passes.
    }];

    let constructor = "mlir::spardial::createSparseEncodingPropagationPass()";
    let dependentDialects = [
        "linalg::LinalgDialect",
        "sparse_tensor::SparseTensorDialect",
        "tensor::TensorDialect"
    ];
}

#endif // SPARDIAL_TRANSFORMS_PASSES
