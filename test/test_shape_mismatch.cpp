// test/test_shape_mismatch.cpp

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SAROps.h"

int main() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::sar::SARDialect>();
    mlir::MLIRContext context(registry);
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::sar::SARDialect>();

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    auto module = builder.create<mlir::ModuleOp>(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto f32 = builder.getF32Type();
    auto tensor2 = mlir::sar::TensorType::get(&context, {2}, f32);
    auto tensor3 = mlir::sar::TensorType::get(&context, {3}, f32);

    auto funcType = builder.getFunctionType({tensor2, tensor3}, {tensor2});
    auto func = builder.create<mlir::func::FuncOp>(loc, "mismatch", funcType);
    func.addEntryBlock();
    auto &block = func.front();
    builder.setInsertionPointToStart(&block);

    auto arg0 = block.getArgument(0);
    auto arg1 = block.getArgument(1);

    auto add = builder.create<mlir::sar::ElemAddOp>(loc, tensor2, arg0, arg1);

    builder.create<mlir::func::ReturnOp>(loc, add.getResult());

    if (mlir::succeeded(mlir::verify(module))) {
        llvm::errs() << "Verification unexpectedly succeeded\n";
        return 1;
    }
    return 0;
}
