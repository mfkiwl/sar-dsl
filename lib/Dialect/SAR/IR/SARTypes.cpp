// lib/Dialect/SAR/IR/SARTypes.cpp

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"

// Include generated type implementations
#define GET_TYPEDEF_CLASSES
#include "Dialect/SAR/IR/SARTypes.cpp.inc"

namespace mlir::sar {

void SARDialect::registerTypes() {
    llvm::outs() << "Register " << getDialectNamespace() << " type\n";
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/SAR/IR/SARTypes.cpp.inc"
    >();
}

// Verify tensor type invariants
::llvm::LogicalResult tensorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType) {
    if (!elementType.isIntOrFloat())
        return emitError() << " Invalid element type ";
    return llvm::success();
}

// Custom tensor type parser
Type tensorType::parse(AsmParser &parser) {
    auto loc = parser.getCurrentLocation();
    SmallVector<int64_t, 4> dimensions;
    Type elementType;

    if (parser.parseLess()             ||  // <
        parser.parseDimensionList(
            dimensions,
            /*allowDynamic=*/true,
            /*withTrailingX=*/true)    ||  // dimensions
        parser.parseType(elementType)  ||  // elementType
        parser.parseGreater()) {           // >

        return Type();
    }

    return get(parser.getContext(), dimensions, elementType);
}

// Custom tensor type printer
void tensorType::print(AsmPrinter &printer) const {
    printer << "<";
    for (int64_t dim : getShape()) {
        if (dim < 0)
            printer << "?" << 'x';
        else
            printer << dim << 'x';
    }
    printer.printType(getElementType());
    printer << ">";
}

}
