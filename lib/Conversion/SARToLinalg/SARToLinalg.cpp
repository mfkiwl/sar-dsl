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

namespace mlir::sar {

    void initSARToLinalgTypeConvert(TypeConverter &typeConverter) {
        typeConverter.addConversion([](tensorType type) {
            return RankedTensorType::get(type.getShape(), type.getElementType());
        });
    }

    void populateSARToLinalgPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns) {
        patterns.add<

        >(typeConverter, patterns.getContext());
    }

} // namespace mlir::sar
