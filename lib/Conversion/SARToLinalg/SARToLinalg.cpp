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

struct ConstOpConverterPattern final : public OpConversionPattern<mlir::sar::ConstOp> {
    using OpConversionPattern<mlir::sar::ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(mlir::sar::ConstOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const final {

        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValueAttr());
        return success();
    }
};

template <typename SarOp>
struct BinaryOpConverterPattern final : public OpConversionPattern<SarOp> {
    using OpConversionPattern<SarOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(SarOp op, typename SarOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const final {

        Location loc = op.getLoc();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();

        if (!llvm::isa<RankedTensorType>(lhs.getType())) {
            llvm::outs() << "lhs type: " << lhs.getType() << "\n";
            return rewriter.notifyMatchFailure(op, "lhs type is not RankedTensorType");
        }
        if (!llvm::isa<RankedTensorType>(rhs.getType())) {
            llvm::outs() << "rhs type: " << rhs.getType() << "\n";
            return rewriter.notifyMatchFailure(op, "rhs type is not RankedTensorType");
        }
        if (!llvm::isa<RankedTensorType>(op.getType())) {
            llvm::outs() << "result type: " << op.getType() << "\n";
            return rewriter.notifyMatchFailure(op, "result type is not RankedTensorType");
        }

        RankedTensorType lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
        RankedTensorType rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
        Type resultType = this->getTypeConverter()->convertType(op.getType());
        RankedTensorType rankedResultType = llvm::dyn_cast<RankedTensorType>(resultType);

        Type elementType = lhsType.getElementType();
        if (elementType != rhsType.getElementType()) {
            return rewriter.notifyMatchFailure(op, "element type mismatch");
        }

        Value output = rewriter.create<tensor::EmptyOp>(
            loc, rankedResultType.getShape(), rankedResultType.getElementType());

        uint64_t rank = rankedResultType.getRank();

        AffineMap lhsMap = rewriter.getMultiDimIdentityMap(rank);
        AffineMap rhsMap = rewriter.getMultiDimIdentityMap(rank);
        AffineMap outputMap = rewriter.getMultiDimIdentityMap(rank);

        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            /*resultTensorTypes=*/TypeRange{rankedResultType},
            /*inputs=*/ValueRange{lhs, rhs},
            /*outputs=*/ValueRange{output},
            /*indexingMaps=*/rewriter.getMultiDimIdentityMap(rank),
            /*iteratorTypes=*/SmallVector<mlir::utils::IteratorType>(rank, mlir::utils::IteratorType::parallel),
            /*doc=*/"",
            /*library_call=*/"",
            [&](OpBuilder &b, Location loc, ValueRange args) {
                Value result;
                Type elementType = args[0].getType();

                if (elementType.isFloat()) {
                    if (std::is_same<SarOp, sar::AddOp>::value) {
                        result = b.create<arith::AddFOp>(loc, args[0], args[1]);
                    } else if (std::is_same<SarOp, sar::SubOp>::value) {
                        result = b.create<arith::SubFOp>(loc, args[0], args[1]);
                    } else if (std::is_same<SarOp, sar::MulOp>::value) {
                        result = b.create<arith::MulFOp>(loc, args[0], args[1]);
                    } else if (std::is_same<SarOp, sar::DivOp>::value) {
                        result = b.create<arith::DivFOp>(loc, args[0], args[1]);
                    }
                } else if (elementType.isInteger()) {
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

}  // namespace

namespace mlir::sar {

    void initSARToLinalgTypeConvert(TypeConverter &typeConverter) {
        typeConverter.addConversion([](mlir::sar::tensorType type) -> Type {
            return RankedTensorType::get(type.getShape(), type.getElementType());
        });

        typeConverter.addConversion([](Type type) -> Type {
            return type;
        });

        typeConverter.addSourceMaterialization(
            [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) 
                -> Value {
                if (inputs.size() != 1) return nullptr;
                return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                    .getResult(0);
            });

        typeConverter.addTargetMaterialization(
            [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) 
                -> Value {
                if (inputs.size() != 1) return nullptr;
                return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                    .getResult(0);
            });
    }

    void populateSARToLinalgPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns) {
        patterns.add<
            ConstOpConverterPattern,
            BinaryOpConverterPattern<mlir::sar::AddOp>,
            BinaryOpConverterPattern<mlir::sar::SubOp>,
            BinaryOpConverterPattern<mlir::sar::MulOp>,
            BinaryOpConverterPattern<mlir::sar::DivOp>
        >(typeConverter, patterns.getContext());
    }

} // namespace mlir::sar
