// lib/Conversion/SARToLinalg/SARToLinalg.cpp

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SAROps.h"
#include "Conversion/SARToLinalg/SARToLinalg.h"

using namespace mlir;

namespace mlir::sar {

    void initSARToLinalgTypeConvert(TypeConverter &typeConverter) {
        typeConverter.addConversion([](tensorType type) {
            return RankedTensorType::get(type.getShape(), type.getElementType());
        });

        typeConverter.addSourceMaterialization(
            [&](OpBuilder &builder, Type resultType, ValueRange inputs,
                Location loc) -> std::optional<Value> {
                if (inputs.size() != 1)
                    return std::nullopt;

                return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                    .getResult(0);
            });
        
        typeConverter.addTargetMaterialization(
            [&](OpBuilder &builder, Type resultType, ValueRange inputs,
                Location loc) -> std::optional<Value> {
                if (inputs.size() != 1)
                    return std::nullopt;

                return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                    .getResult(0);
            });
    }

    void populateSARToLinalgPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
    }

} // namespace mlir::sar
