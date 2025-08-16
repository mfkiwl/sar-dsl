#include <memory>
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SAROps.h"
#include "Conversion/SARToLinalg/SARToLinalg.h"

#define DEBUG_TYPE "convert-sar-to-linalg"

namespace mlir::sar {

#define GEN_PASS_DEF_CONVERTSARTOLINALGPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir::sar

using namespace ::mlir;
using namespace ::mlir::sar;

struct ConvertSARToLinalgPass
    : public mlir::sar::impl::ConvertSARToLinalgPassBase<ConvertSARToLinalgPass> {
    void runOnOperation() override;
};

void configSARToLinalgTarget(TypeConverter &typeConverter, ConversionTarget &target) {
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<arith::ArithDialect>();

    target.addIllegalDialect<sar::SARDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               llvm::all_of(op.getArgumentTypes(), [&](Type type) { return typeConverter.isLegal(type); }) &&
               llvm::all_of(op.getResultTypes(), [&](Type type) { return typeConverter.isLegal(type); });
    });

    target.addDynamicallyLegalOp<func::ReturnOp>([&typeConverter](func::ReturnOp op) {
        return llvm::all_of(op.getOperandTypes(), [&](Type type) {
            return typeConverter.isLegal(type);
        });
    });
}

void ConvertSARToLinalgPass::runOnOperation() {
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
    auto model = getOperation();
    TypeConverter type_convert;
    initSARToLinalgTypeConvert(type_convert);
    RewritePatternSet patterns(&getContext());
    populateSARToLinalgPatterns(type_convert, patterns);
    ConversionTarget target(getContext());
    configSARToLinalgTarget(type_convert, target);
    if (failed(applyPartialConversion(model, target, std::move(patterns))))
        signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out {0}\n", getPassName()));
}

void mlir::sar::registerSARPassPipelines() {
    static PassPipelineRegistration<> sarToLinalgPipeline(
        "sar-to-linalg-pipeline",
        "SAR to Linalg conversion pipeline",
        [](OpPassManager &pm) { pm.addPass(mlir::sar::createConvertSARToLinalgPass()); });
}

mlir::LogicalResult mlir::sar::runSARToLinalgPipeline(mlir::ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(mlir::sar::createConvertSARToLinalgPass());
    return pm.run(module);
}
