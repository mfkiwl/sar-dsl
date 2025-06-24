// include/Conversion/Passes.td

#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD
include "mlir/Pass/PassBase.td"

def ConvertSARToLinalgPass : Pass<"convert-sar-to-linalg", "mlir::ModuleOp"> {
    let summary = "Convert SAR dialect to Linalg dialect";
    let description = "This pass converts SAR dialect operations to equivalent Linalg dialect operations.";

    let dependentDialects = [
        "::mlir::sar::SARDialect",
        "::mlir::tensor::TensorDialect",
        "::mlir::linalg::LinalgDialect"
    ];
}

#endif
