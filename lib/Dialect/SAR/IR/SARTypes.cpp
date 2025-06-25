// lib/Dialect/SAR/IR/SARTypes.cpp

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/SAR/IR/SARTypes.cpp.inc"

namespace mlir::sar {

void SARDialect::registerTypes() {
    // llvm::outs() << "Register " << getDialectNamespace() << " type\n";
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/SAR/IR/SARTypes.cpp.inc"
    >();
}

::llvm::LogicalResult tensorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType) {
    if (!elementType.isIntOrFloat()) {
        return emitError() << "tensor element type must be integer or float";
    }
    return ::mlir::success();
}

Type tensorType::parse(AsmParser &parser) {
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

    return tensorType::get(parser.getContext(), dimensions, elementType);
}

void tensorType::print(AsmPrinter &printer) const {
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
