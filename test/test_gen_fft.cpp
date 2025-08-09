// test/test_gen_fft.cpp

#include <fstream>
#include <filesystem>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
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
    registry.insert<mlir::BuiltinDialect,
                    mlir::func::FuncDialect,
                    mlir::sar::SARDialect>();
    mlir::MLIRContext context(registry);
    context.loadDialect<mlir::BuiltinDialect,
                        mlir::func::FuncDialect,
                        mlir::sar::SARDialect>();

    // Create OpBuilder
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Create module
    auto module = builder.create<mlir::ModuleOp>(loc, "sar_fft_module");
    builder.setInsertionPointToStart(module.getBody());

    // Define types
    auto f32 = builder.getF32Type();
    mlir::SmallVector<int64_t> tensor_shape = {8, 4, 2};
    auto tensor_type = mlir::sar::TensorType::get(&context, tensor_shape, f32);
    mlir::SmallVector<int64_t> matrix_shape = {4, 4};
    auto matrix_type = mlir::sar::MatrixType::get(&context, matrix_shape, f32);

    // Create function type: (tensor, matrix) -> multiple results (Tensor x6, Matrix x6)
    auto functionType = builder.getFunctionType(
        {tensor_type, matrix_type},
        {tensor_type, tensor_type, tensor_type, tensor_type, tensor_type, tensor_type,
         matrix_type, matrix_type, matrix_type, matrix_type, matrix_type, matrix_type}
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
    auto tensorArg = entryBlock.getArgument(0); // tensor
    auto matrixArg = entryBlock.getArgument(1); // matrix

    // Apply FFT operations on Tensor: FFTnDim, IFFTNDim, FFTDimx (0,1,2), IFFTDimx (0,1,2)
    auto t_fft_nd = builder.create<mlir::sar::FFTnDimOp>(loc, tensor_type, tensorArg);
    auto t_ifft_nd = builder.create<mlir::sar::IFFTnDimOp>(loc, tensor_type, tensorArg);
    auto t_fft_d0 = builder.create<mlir::sar::FFTDimxOp>(loc, tensor_type, tensorArg, builder.getI64IntegerAttr(0));
    auto t_fft_d1 = builder.create<mlir::sar::FFTDimxOp>(loc, tensor_type, tensorArg, builder.getI64IntegerAttr(1));
    // removed: t_fft_d2 (not required by test spec)
    auto t_ifft_d0 = builder.create<mlir::sar::IFFTDimxOp>(loc, tensor_type, tensorArg, builder.getI64IntegerAttr(0));
    auto t_ifft_d1 = builder.create<mlir::sar::IFFTDimxOp>(loc, tensor_type, tensorArg, builder.getI64IntegerAttr(1));
    // removed: t_ifft_d2 (not required by test spec)

    // Apply FFT operations on Matrix: FFTnDim, IFFTNDim, FFTDimx (0,1), IFFTDimx (0,1)
    auto m_fft_nd = builder.create<mlir::sar::FFTnDimOp>(loc, matrix_type, matrixArg);
    auto m_ifft_nd = builder.create<mlir::sar::IFFTnDimOp>(loc, matrix_type, matrixArg);
    auto m_fft_d0 = builder.create<mlir::sar::FFTDimxOp>(loc, matrix_type, matrixArg, builder.getI64IntegerAttr(0));
    auto m_fft_d1 = builder.create<mlir::sar::FFTDimxOp>(loc, matrix_type, matrixArg, builder.getI64IntegerAttr(1));
    auto m_ifft_d0 = builder.create<mlir::sar::IFFTDimxOp>(loc, matrix_type, matrixArg, builder.getI64IntegerAttr(0));
    auto m_ifft_d1 = builder.create<mlir::sar::IFFTDimxOp>(loc, matrix_type, matrixArg, builder.getI64IntegerAttr(1));

    // Add return operation with all results
    builder.create<mlir::func::ReturnOp>(
        loc,
        mlir::ValueRange{
            t_fft_nd.getResult(), t_ifft_nd.getResult(), t_fft_d0.getResult(), t_fft_d1.getResult(),
            t_ifft_d0.getResult(), t_ifft_d1.getResult(),
            m_fft_nd.getResult(), m_ifft_nd.getResult(), m_fft_d0.getResult(), m_fft_d1.getResult(), m_ifft_d0.getResult(), m_ifft_d1.getResult()
        }
    );

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
        outputFile << "\n";
        outputFile.close();
        llvm::outs() << "Successfully generated MLIR file: " << outFile << "\n";
    }
    return 0;
}
