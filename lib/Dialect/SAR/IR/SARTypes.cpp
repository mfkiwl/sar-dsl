// lib/Dialect/SAR/IR/SARTypes.cpp

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/SAR/IR/SARTypes.cpp.inc"

namespace mlir::sar {

void SARDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/SAR/IR/SARTypes.cpp.inc"
    >();
}

::mlir::LogicalResult tensorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType) {
    if (!elementType.isIntOrFloat()) {
        return emitError() << "tensor element type must be integer or float";
    }
    return ::mlir::success();
}

::mlir::Type tensorType::parse(::mlir::AsmParser &parser) {
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

    return get(parser.getContext(), dimensions, elementType);
}

void tensorType::print(::mlir::AsmPrinter &printer) const {
    printer << "<";
    for (auto dim : getShape()) {
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
