// lib/Dialect/SAR/IR/SARTypes.cpp

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/SAR/IR/SARTypes.cpp.inc"

namespace mlir::sar {

// Register types for the SAR dialect
void SARDialect::registerTypes() {
    llvm::outs() << "Register " << getDialectNamespace() << " type\n";
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/SAR/IR/SARTypes.cpp.inc"
    >();
}

// Verify the tensor type
::mlir::LogicalResult mlir::sar::tensorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType) {
    // Check if element type is integer or float
    if (!elementType.isIntOrFloat()) {
        return emitError() << "tensor element type must be integer or float";
    }
    return ::mlir::success();
}

// Parse the tensor type from assembly format
::mlir::Type mlir::sar::tensorType::parse(::mlir::AsmParser &parser) {
    ::llvm::SmallVector<int64_t, 4> shape;
    ::mlir::Type elementType;
    
    // Parse the type format: <shape x elementType>
    if (parser.parseLess() ||
        parser.parseDimensionList(shape, /*allowDynamic=*/true) ||
        parser.parseType(elementType) ||
        parser.parseGreater()) {
        return {};
    }
    
    return get(parser.getContext(), shape, elementType);
}

// Print the tensor type to assembly format
void mlir::sar::tensorType::print(::mlir::AsmPrinter &printer) const {
    printer << "<";
    auto shape = getShape();
    for (int i = 0; i < shape.size(); ++i) {
        if (i > 0)
            printer << "x";
        if (shape[i] == ShapedType::kDynamic) {
            printer << "?";
        } else {
            printer << shape[i];
        }
    }
    printer << "x";
    printer.printType(getElementType());
    printer << ">";
}

} // namespace mlir::sar
