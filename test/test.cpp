#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SAROps.h"

int main() {
    /******************************************************************
     * Initialize MLIR context and register SAR dialect
     ******************************************************************/

    // Create a dialect registry to hold all MLIR dialects
    mlir::DialectRegistry registry;

    // Register the SAR dialect with the registry
    registry.insert<mlir::sar::SARDialect>();

    // Create MLIR context using our registry
    mlir::MLIRContext context(registry);

    // Load the SAR dialect from the context
    auto* dialect = context.getOrLoadDialect<mlir::sar::SARDialect>();

    // Call dialect-specific greeting method
    dialect->sayHello();

    /******************************************************************
     * Demonstrate SAR tensor type creation and inspection
     ******************************************************************/
    
    // Create a static SAR tensor type: 1x2x3 f32
    mlir::sar::tensorType sar_tensor = mlir::sar::tensorType::get(
        &context, {1, 2, 3}, mlir::Float32Type::get(&context));
    llvm::outs() << "SAR tensor type: ";
    sar_tensor.dump();  // Print type representation

    // Create a dynamic SAR tensor type: ?x2x3 f32
    mlir::sar::tensorType dynamic_sar_tensor = mlir::sar::tensorType::get(
        &context, {mlir::ShapedType::kDynamic, 2, 3}, 
        mlir::Float32Type::get(&context));
    llvm::outs() << "Dynamic SAR tensor type: ";
    dynamic_sar_tensor.dump();

    /******************************************************************
     * Build SAR computation graph using MLIR operations
     ******************************************************************/
    
    // Create an OpBuilder for constructing operations
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();  // Default location for operations
    
    // Create top-level module operation named "SAR"
    auto module = builder.create<mlir::ModuleOp>(loc, "SAR");
    
    // Set insertion point to module body for subsequent operations
    builder.setInsertionPointToStart(module.getBody());
    
    // Get float32 type for element specification
    auto f32 = mlir::Float32Type::get(&context);
    
    // Define tensor shape as 2x2
    auto shape = mlir::SmallVector<int64_t>({2, 2});
    
    // Create constant values (4 elements each with value 1.0 and 2.0)
    auto const_value1 = mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat(1.0f));
    auto const_value2 = mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat(2.0f));
    
    // Create SAR tensor type for constants
    auto tensor_type = mlir::sar::tensorType::get(&context, shape, f32);
    
    // Create constant operations with dense attribute values
    auto const1 = builder.create<mlir::sar::ConstOp>(
        loc, 
        tensor_type, 
        mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get(shape, f32), 
            const_value1
        )
    );
    auto const2 = builder.create<mlir::sar::ConstOp>(
        loc, 
        tensor_type, 
        mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get(shape, f32), 
            const_value2
        )
    );
    
    // Build computation graph: (const1 + const2) * const2 - const1
    auto add = builder.create<mlir::sar::AddOp>(loc, const1, const2);
    auto mul = builder.create<mlir::sar::MulOp>(loc, add, const2);
    auto sub = builder.create<mlir::sar::SubOp>(loc, mul, const1);
    
    // Add print operation to output computation result
    builder.create<mlir::sar::PrintOp>(loc, sub);

    /******************************************************************
     * Verify and output the constructed module
     ******************************************************************/
    
    // Verify module structure and operation validity
    if (mlir::failed(module.verify())) {
        llvm::errs() << "Module verification failed!\n";
        return 1;  // Exit with error code if verification fails
    }

    // Dump the entire module to stdout
    llvm::outs() << "\nFinal SAR module:\n";
    module.dump();

    return 0;
}
