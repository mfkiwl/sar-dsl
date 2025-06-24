// tools/sar-opt.cpp

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Conversion/Passes.h"
#include "Dialect/SAR/IR/SARDialect.h"

int main(int argc, char **argv) {
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registerAllDialects(registry);
    registerAllExtensions(registry);
    
    // Register SAR dialect and related dialects
    registry.insert<mlir::sar::SARDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    
    // Register SAR passes
    mlir::sar::registerSARConversionPasses();
    
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "SAR Optimizer Driver", registry));
}
