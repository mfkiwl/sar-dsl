// lib/Conversion/SARToLinalg/SARToLinalg.cpp

#include <memory>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SAROps.h"
#include "Conversion/SARToLinalg/SARToLinalg.h"

using namespace mlir;

namespace {

// Convert sar::ConstOp to arith::ConstantOp
struct ConstOpConverterPattern final : public OpConversionPattern<mlir::sar::ConstOp> {
    using OpConversionPattern<mlir::sar::ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(mlir::sar::ConstOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {

        // Convert SAR tensor type to standard tensor type
        Type newType = getTypeConverter()->convertType(op.getResult().getType());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newType, op.getValueAttr());
        return success();
    }
};

// Generic pattern for converting SAR binary operations (add/sub/mul/div)
// to equivalent linalg.generic operations
template <typename SarOp>
struct BinaryOpConverterPattern final : public OpConversionPattern<SarOp> {
    using OpConversionPattern<SarOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(SarOp op, typename SarOp::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {

        Location loc = op.getLoc();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();

        // Convert result type to standard tensor type
        Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
        auto rankedResultType = llvm::dyn_cast<RankedTensorType>(resultType);
        if (!rankedResultType)
            return rewriter.notifyMatchFailure(op, "result type is not RankedTensorType");

        // Check input operand types
        auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
        auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
        if (!lhsType || !rhsType)
            return rewriter.notifyMatchFailure(op, "operand type is not RankedTensorType");

        // Verify element type consistency
        Type elementType = lhsType.getElementType();
        if (elementType != rhsType.getElementType())
            return rewriter.notifyMatchFailure(op, "element type mismatch");

        // Create empty output tensor for linalg operation
        Value output = rewriter.create<tensor::EmptyOp>(
            loc, rankedResultType.getShape(), rankedResultType.getElementType());

        uint64_t rank = rankedResultType.getRank();

        // Create identity maps for input/output indexing
        AffineMap lhsMap = rewriter.getMultiDimIdentityMap(rank);
        AffineMap rhsMap = rewriter.getMultiDimIdentityMap(rank);
        AffineMap outputMap = rewriter.getMultiDimIdentityMap(rank);

        // Generate linalg.generic operation for element-wise computation
        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            /*resultTensorTypes=*/TypeRange{rankedResultType},
            /*inputs=*/ValueRange{lhs, rhs},
            /*outputs=*/ValueRange{output},
            /*indexingMaps=*/ArrayRef<AffineMap>{lhsMap, rhsMap, outputMap},
            /*iteratorTypes=*/SmallVector<mlir::utils::IteratorType>(rank, mlir::utils::IteratorType::parallel),
            /*doc=*/"",
            /*library_call=*/"",
            [&](OpBuilder &b, Location loc, ValueRange args) {
                // Element-wise operation dispatch based on SAR op type
                Value result;
                Type elemType = args[0].getType();

                if (elemType.isFloat()) {
                    if (std::is_same<SarOp, sar::AddOp>::value) {
                        result = b.create<arith::AddFOp>(loc, args[0], args[1]);
                    } else if (std::is_same<SarOp, sar::SubOp>::value) {
                        result = b.create<arith::SubFOp>(loc, args[0], args[1]);
                    } else if (std::is_same<SarOp, sar::MulOp>::value) {
                        result = b.create<arith::MulFOp>(loc, args[0], args[1]);
                    } else if (std::is_same<SarOp, sar::DivOp>::value) {
                        result = b.create<arith::DivFOp>(loc, args[0], args[1]);
                    }
                } else if (elemType.isInteger()) {
                    if (std::is_same<SarOp, sar::AddOp>::value) {
                        result = b.create<arith::AddIOp>(loc, args[0], args[1]);
                    } else if (std::is_same<SarOp, sar::SubOp>::value) {
                        result = b.create<arith::SubIOp>(loc, args[0], args[1]);
                    } else if (std::is_same<SarOp, sar::MulOp>::value) {
                        result = b.create<arith::MulIOp>(loc, args[0], args[1]);
                    } else if (std::is_same<SarOp, sar::DivOp>::value) {
                        result = b.create<arith::DivSIOp>(loc, args[0], args[1]);
                    }
                }

                if (!result) {
                    emitError(loc, "unsupported binary operation");
                    return;
                }

                b.create<linalg::YieldOp>(loc, result);
            });

        rewriter.replaceOp(op, genericOp.getResults());
        return success();
    }
};

// Convert function signatures and body types from SAR to standard types
struct FuncSignatureConversion : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {

        // Convert function argument types
        TypeConverter::SignatureConversion signatureConverter(op.getFunctionType().getNumInputs());
        auto origFuncType = op.getFunctionType();
        for (const auto &arg : llvm::enumerate(origFuncType.getInputs())) {
            Type convertedType = getTypeConverter()->convertType(arg.value());
            if (!convertedType)
                return rewriter.notifyMatchFailure(op, "argument type conversion failed");
            signatureConverter.addInputs(arg.index(), convertedType);
        }

        // Convert function result types
        SmallVector<Type, 1> convertedResultTypes;
        if (failed(getTypeConverter()->convertTypes(origFuncType.getResults(), convertedResultTypes)))
            return rewriter.notifyMatchFailure(op, "result type conversion failed");

        // Create new function with converted signature
        auto newFuncType = rewriter.getFunctionType(signatureConverter.getConvertedTypes(), convertedResultTypes);
        auto newFunc = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), newFuncType);

        // Propagate type conversions through function body
        rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
        Block* entryBlock = &newFunc.getBody().front();
        rewriter.applySignatureConversion(entryBlock, signatureConverter);

        // Insert casts for block arguments that changed type
        for (auto [blockArg, origType] : llvm::zip(entryBlock->getArguments(), origFuncType.getInputs())) {
            if (blockArg.getType() != origType) {
                auto cast = rewriter.create<UnrealizedConversionCastOp>(
                    blockArg.getLoc(), origType, blockArg);
                blockArg.replaceAllUsesExcept(cast.getResult(0), cast);
            }
        }

        // Update all SAR tensor types within function body
        for (Block &block : newFunc.getBody()) {
            for (Operation &op0 : llvm::make_early_inc_range(block.getOperations())) {
                for (auto result : op0.getResults()) {
                    auto sarType = llvm::dyn_cast<RankedTensorType>(result.getType());
                    if (!sarType)
                        continue;
                    Type t = getTypeConverter()->convertType(sarType);
                    if (!t || t == sarType)
                        continue;
                    auto cast = rewriter.create<UnrealizedConversionCastOp>(
                        result.getLoc(), t, result);
                    result.replaceAllUsesExcept(cast.getResult(0), cast);
                }
            }
        }

        rewriter.replaceOp(op, newFunc.getOperation()->getResults());
        return success();
    }
};

// Update return op to handle converted types
struct ReturnOpConverter : public OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {

        // Simply forward converted operands
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

}  // namespace

namespace mlir::sar {

// Initialize type conversions for SAR dialect
void initSARToLinalgTypeConvert(TypeConverter &typeConverter) {

    // Convert SAR tensor types to standard ranked tensors
    typeConverter.addConversion([](mlir::sar::TensorType type) -> Type {
        return RankedTensorType::get(type.getShape(), type.getElementType());
    });

    typeConverter.addConversion([](RankedTensorType type) -> Type { return type; });
    typeConverter.addConversion([](UnrankedTensorType type) -> Type { return type; });
}

// Populate conversion patterns for SAR-to-Linalg lowering
void populateSARToLinalgPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns) {
    patterns.add<
        ConstOpConverterPattern,
        BinaryOpConverterPattern<mlir::sar::AddOp>,
        BinaryOpConverterPattern<mlir::sar::SubOp>,
        BinaryOpConverterPattern<mlir::sar::MulOp>,
        BinaryOpConverterPattern<mlir::sar::DivOp>,
        FuncSignatureConversion,
        ReturnOpConverter
    >(typeConverter, patterns.getContext());
}

} // namespace mlir::sar
