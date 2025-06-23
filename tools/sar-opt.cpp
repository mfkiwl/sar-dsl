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
    // Register all MLIR passes
    mlir::registerAllPasses();
    
    // Create dialect registry
    mlir::DialectRegistry registry;
    
    // Register all core dialects and extensions
    registerAllDialects(registry);
    registerAllExtensions(registry);
    
    // Register custom SAR dialect
    registry.insert<mlir::sar::SARDialect>();
    
    // Register custom SAR conversion passes
    mlir::sar::registerSARConversionPasses();
    
    // Run the MLIR optimization driver
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "SAR Optimizer Driver", registry));
}
