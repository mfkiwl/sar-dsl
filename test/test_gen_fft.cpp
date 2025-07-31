// test/test_gen_fft.cpp

#include <fstream>
#include <filesystem>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SAROps.h"

int main(int argc, char **argv) {
    std::string outFile;
    // Simple parsing for -o argument
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-o" && i + 1 < argc) {
            outFile = argv[++i];
        }
    }

    // Initialize MLIR context and register dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::sar::SARDialect>();
    registry.insert<mlir::func::FuncDialect>();
    mlir::MLIRContext context(registry);
    context.loadDialect<mlir::sar::SARDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    // Create OpBuilder
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Create module
    auto module = builder.create<mlir::ModuleOp>(loc, "sar_fft_module");
    builder.setInsertionPointToStart(module.getBody());

    // Define types
    auto f32 = builder.getF32Type();
    mlir::SmallVector<int64_t> vector_shape = {8};
    auto vector_type = mlir::sar::VectorType::get(&context, vector_shape, f32);
    mlir::SmallVector<int64_t> matrix_shape = {4, 4};
    auto matrix_type = mlir::sar::MatrixType::get(&context, matrix_shape, f32);

    // Create function type: (vector, matrix) -> (vector, matrix)
    auto functionType = builder.getFunctionType(
        {vector_type, matrix_type}, 
        {vector_type, matrix_type}
    );

    // Create function
    auto func = builder.create<mlir::func::FuncOp>(
        loc, 
        "forward",
        functionType
    );
    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Get function arguments
    auto arg0 = entryBlock.getArgument(0); // vector
    auto arg1 = entryBlock.getArgument(1); // matrix

    // Apply FFT operations
    auto fft1d = builder.create<mlir::sar::FFT1dOp>(loc, vector_type, arg0);
    auto fft2d0 = builder.create<mlir::sar::FFT2d0Op>(loc, matrix_type, arg1);
    auto fft2d1 = builder.create<mlir::sar::FFT2d1Op>(loc, matrix_type, fft2d0);

    // Add return operation
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{fft1d.getResult(), fft2d1.getResult()});

    // Verify module
    if (mlir::failed(module.verify())) {
        llvm::errs() << "Module verification failed!\n";
        return 1;
    }

    // Determine output method
    if (outFile.empty()) {
        // Print directly to terminal if no -o specified
        mlir::OpPrintingFlags flags;
        module.print(llvm::outs(), flags);
        llvm::outs() << "\n";
    } else {
        // -o is specified

        // Make sure the output directory exists
        std::filesystem::path outputPath(outFile);
        std::filesystem::path parentDir = outputPath.parent_path();
        if (!parentDir.empty() && !std::filesystem::exists(parentDir)) {
            if (!std::filesystem::create_directories(parentDir)) {
                llvm::errs() << "Failed to create directory: " << parentDir << "\n";
                return 1;
            }
        }

        // Output to file
        std::error_code ec;
        llvm::raw_fd_ostream outputFile(outFile, ec);
        if (ec) {
            llvm::errs() << "Failed to open output file: " << ec.message() << "\n";
            return 1;
        }
        mlir::OpPrintingFlags flags;
        module.print(outputFile, flags);
        outputFile << "\n";  // Ensure newline at end of file
        outputFile.close();
        llvm::outs() << "Successfully generated MLIR file: " << outFile << "\n";
    }
    return 0;
} 