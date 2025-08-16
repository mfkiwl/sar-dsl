#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/SAR/IR/SARTypes.cpp.inc"

namespace mlir::sar {

static bool isSupportedElementType(Type elementType) {
    if (elementType.isIntOrFloat())
        return true;
    if (auto ct = llvm::dyn_cast<mlir::ComplexType>(elementType))
        return llvm::isa<mlir::FloatType>(ct.getElementType());
    return false;
}

void SARDialect::registerTypes() {
    // llvm::outs() << "Register " << getDialectNamespace() << " type\n";
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/SAR/IR/SARTypes.cpp.inc"
    >();
}

::llvm::LogicalResult TensorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType) {
    if (!isSupportedElementType(elementType)) {
        return emitError() << "tensor element type must be integer, float, or complex of float";
    }
    return ::mlir::success();
}

Type TensorType::parse(AsmParser &parser) {
    if (parser.parseLess()) 
        return Type();

    SmallVector<int64_t, 4> dimensions;
    if (parser.parseDimensionList(dimensions, /*allowDynamic=*/true, /*withTrailingX=*/true))
        return Type();

    Type elementType;
    if (parser.parseType(elementType)) 
        return Type();

    if (parser.parseGreater()) 
        return Type();

    return TensorType::get(parser.getContext(), dimensions, elementType);
}

void TensorType::print(AsmPrinter &printer) const {
    printer << "<";
    for (int64_t dim : getShape()) {
        if (dim < 0) {
            printer << "?" << "x";
        } else {
            printer << dim << "x";
        }
    }
    printer.printType(getElementType());
    printer << ">";
}

::llvm::LogicalResult MatrixType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType) {
    if (shape.size() != 2) {
        return emitError() << "matrix must have exactly 2 dimensions";
    }
    if (!isSupportedElementType(elementType)) {
        return emitError() << "matrix element type must be integer, float, or complex of float";
    }
    return ::mlir::success();
}

Type MatrixType::parse(AsmParser &parser) {
    if (parser.parseLess()) 
        return Type();

    SmallVector<int64_t, 2> dimensions;
    if (parser.parseDimensionList(dimensions, /*allowDynamic=*/true, /*withTrailingX=*/true))
        return Type();

    Type elementType;
    if (parser.parseType(elementType)) 
        return Type();

    if (parser.parseGreater()) 
        return Type();

    return MatrixType::get(parser.getContext(), dimensions, elementType);
}

void MatrixType::print(AsmPrinter &printer) const {
    printer << "<";
    for (int64_t dim : getShape()) {
        if (dim < 0) {
            printer << "?" << "x";
        } else {
            printer << dim << "x";
        }
    }
    printer.printType(getElementType());
    printer << ">";
}

::llvm::LogicalResult VectorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType) {
    if (shape.size() != 1) {
        return emitError() << "vector must have exactly 1 dimension";
    }
    if (!isSupportedElementType(elementType)) {
        return emitError() << "vector element type must be integer, float, or complex of float";
    }
    return ::mlir::success();
}

Type VectorType::parse(AsmParser &parser) {
    if (parser.parseLess()) 
        return Type();

    SmallVector<int64_t, 1> dimensions;
    if (parser.parseDimensionList(dimensions, /*allowDynamic=*/true, /*withTrailingX=*/true))
        return Type();

    Type elementType;
    if (parser.parseType(elementType)) 
        return Type();

    if (parser.parseGreater()) 
        return Type();

    return VectorType::get(parser.getContext(), dimensions, elementType);
}

void VectorType::print(AsmPrinter &printer) const {
    printer << "<";
    for (int64_t dim : getShape()) {
        if (dim < 0) {
            printer << "?" << "x";
        } else {
            printer << dim << "x";
        }
    }
    printer.printType(getElementType());
    printer << ">";
}

} // namespace mlir::sar
