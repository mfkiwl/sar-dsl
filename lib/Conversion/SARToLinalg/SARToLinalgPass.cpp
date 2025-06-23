// lib/Conversion/SARToLinalg/SARToLinalgPass.cpp

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

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

struct SARToLinalgPassPass
    : public mlir::sar::impl::ConvertSARToLinalgPassBase<
          SARToLinalgPassPass> {
  void runOnOperation() override;
};

void configSARToLinalgTarget(ConversionTarget& target) {
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
        for (auto type : op->getOperandTypes()) {
            if (isa<::mlir::sar::tensorType>(type))
                return false;
        }
        return true;
    });
}

void SARToLinalgPassPass::runOnOperation() {
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
    auto model = getOperation();
    TypeConverter type_convert;
    initSARToLinalgTypeConvert(type_convert);
    RewritePatternSet patterns(&getContext());
    populateSARToLinalgPatterns(type_convert, patterns);
    ConversionTarget target(getContext());
    configSARToLinalgTarget(target);
    
    if (failed(applyPartialConversion(model, target, std::move(patterns))))
        signalPassFailure();
    
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}
