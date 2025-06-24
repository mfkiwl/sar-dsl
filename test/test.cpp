// test/test.cpp

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SAROps.h"
#include "mlir/Support/LLVM.h"

int main() {
    // /******************************************************************
    //  * Initialize MLIR context and register dialects
    //  ******************************************************************/

    // // Create a dialect registry to hold all MLIR dialects
    // mlir::DialectRegistry registry;

    // // Register SAR dialect with registry
    // registry.insert<mlir::sar::SARDialect>();

    // // Register Func dialect with registry
    // registry.insert<mlir::func::FuncDialect>();

    // // Create MLIR context using registry
    // mlir::MLIRContext context(registry);

    // // Load SAR dialect from context
    // auto* sarDialect = context.getOrLoadDialect<mlir::sar::SARDialect>();

    // // Load Func dialect from context
    // auto* funcDialect = context.getOrLoadDialect<mlir::func::FuncDialect>();

    // // Call dialect-specific greeting method
    // sarDialect->sayHello();

    // /******************************************************************
    //  * Demonstrate SAR tensor type creation and inspection
    //  ******************************************************************/

    // // Create a static SAR tensor type: 1x2x3 f32
    // mlir::sar::tensorType sar_tensor = mlir::sar::tensorType::get(
    //     &context, {1, 2, 3}, mlir::Float32Type::get(&context));
    // llvm::outs() << "SAR tensor type: ";
    // sar_tensor.dump();

    // // Create a dynamic SAR tensor type: ?x2x3 f32
    // mlir::sar::tensorType dynamic_sar_tensor = mlir::sar::tensorType::get(
    //     &context, {mlir::ShapedType::kDynamic, 2, 3}, 
    //     mlir::Float32Type::get(&context));
    // llvm::outs() << "Dynamic SAR tensor type: ";
    // dynamic_sar_tensor.dump();

    // /******************************************************************
    //  * Build SAR computation module
    //  ******************************************************************/

    // // Create an OpBuilder for constructing operations
    // mlir::OpBuilder builder(&context);
    // auto loc = builder.getUnknownLoc();  // Default location for operations

    // // Create top-level module operation
    // auto module = builder.create<mlir::ModuleOp>(loc, "sar_module");

    // // Set insertion point to module body for subsequent operations
    // builder.setInsertionPointToStart(module.getBody());

    // // Get float32 type for element specification
    // auto f32 = builder.getF32Type();

    // // Define tensor shape as 2x2 (total 4 elements)
    // auto shape = mlir::SmallVector<int64_t>({2, 2});
    // auto tensor_type = mlir::sar::tensorType::get(&context, shape, f32);

    // // Create constant values (4 elements each with value 1.0 and 2.0)
    // auto const_value1 = mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat(1.0f));
    // auto const_value2 = mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat(2.0f));

    // /******************************************************************
    //  * Create a function for SAR computation
    //  ******************************************************************/

    // // Define function type: (tensor, tensor) -> tensor
    // auto functionType = mlir::FunctionType::get(
    //     &context, 
    //     {tensor_type, tensor_type},  // Input types
    //     {tensor_type}                // Return type
    // );

    // // Create function operation
    // auto func = builder.create<mlir::func::FuncOp>(
    //     loc, 
    //     "sar_computation",  // Function name
    //     functionType
    // );

    // // Create function entry block
    // auto &entryBlock = *func.addEntryBlock();
    // builder.setInsertionPointToStart(&entryBlock);

    // // Get function arguments
    // auto arg0 = entryBlock.getArgument(0);  // First input tensor
    // auto arg1 = entryBlock.getArgument(1);  // Second input tensor

    // /******************************************************************
    //  * Build computation inside function using function arguments
    //  ******************************************************************/

    // // Create constant operations with dense attribute values
    // auto const1 = builder.create<mlir::sar::ConstOp>(
    //     loc, 
    //     tensor_type, 
    //     mlir::DenseElementsAttr::get(
    //         mlir::RankedTensorType::get(shape, f32), 
    //         const_value1
    //     )
    // );
    // auto const2 = builder.create<mlir::sar::ConstOp>(
    //     loc, 
    //     tensor_type, 
    //     mlir::DenseElementsAttr::get(
    //         mlir::RankedTensorType::get(shape, f32), 
    //         const_value2
    //     )
    // );

    // // Build computation graph: (arg0 + const1) - (arg1 * const2)
    // auto add = builder.create<mlir::sar::AddOp>(loc, arg0, const1);
    // auto mul = builder.create<mlir::sar::MulOp>(loc, arg1, const2);
    // auto result = builder.create<mlir::sar::SubOp>(loc, add, mul);

    // // Add print operation inside function
    // builder.create<mlir::sar::PrintOp>(loc, result);

    // // Return computed result
    // builder.create<mlir::func::ReturnOp>(loc, result.getResult());

    // /******************************************************************
    //  * Verify and output constructed module
    //  ******************************************************************/

    // // Verify module structure and operation validity
    // if (mlir::failed(module.verify())) {
    //     llvm::errs() << "Module verification failed!\n";
    //     return 1;
    // }

    // // Dump entire module to stdout
    // llvm::outs() << "Final SAR module:\n";

    // mlir::OpPrintingFlags flags;
    // // flags.printGenericOpForm();
    // module.print(llvm::outs(), flags);

    return 0;
}
