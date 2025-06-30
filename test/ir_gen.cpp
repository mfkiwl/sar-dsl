// test/ir_gen.cpp

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
    auto module = builder.create<mlir::ModuleOp>(loc, "sar_module");
    builder.setInsertionPointToStart(module.getBody());

    // Define 3x3x3 tensor type
    auto f32 = builder.getF32Type();
    mlir::SmallVector<int64_t> shape = {3, 3, 3};
    auto tensor_type = mlir::sar::tensorType::get(&context, shape, f32);

    // Create function type: (tensor, tensor) -> tensor
    auto functionType = builder.getFunctionType(
        {tensor_type, tensor_type}, 
        {tensor_type}
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
    auto arg0 = entryBlock.getArgument(0);
    auto arg1 = entryBlock.getArgument(1);

    // Create constant values
    auto createConstTensor = [&](float value) -> mlir::Value {
        // Create standard tensor type for attributes
        auto std_tensor_type = mlir::RankedTensorType::get(shape, f32);
        
        // Create fill values
        mlir::SmallVector<llvm::APFloat> values(27, llvm::APFloat(value));
        auto data_attr = mlir::DenseElementsAttr::get(std_tensor_type, values);
        
        // Create SAR constant operation
        return builder.create<mlir::sar::ConstOp>(
            loc, 
            tensor_type, 
            data_attr
        ).getResult();
    };

    auto const1 = createConstTensor(1.0f);
    auto const2 = createConstTensor(2.0f);

    // Build computation graph: (arg0 + const1) - (arg1 * const2)
    auto add = builder.create<mlir::sar::AddOp>(loc, tensor_type, arg0, const1);
    auto mul = builder.create<mlir::sar::MulOp>(loc, tensor_type, arg1, const2);
    auto result = builder.create<mlir::sar::SubOp>(loc, tensor_type, add, mul);

    // Add return operation
    builder.create<mlir::func::ReturnOp>(loc, result.getResult());

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
