// lib/Dialect/SAR/IR/SAROps.cpp

#include "mlir/IR/Types.h"
#include "llvm/Support/Casting.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include "Dialect/SAR/IR/SAROps.h"

#define GET_OP_CLASSES
#include "Dialect/SAR/IR/SAROps.cpp.inc"

namespace mlir::sar {

void SARDialect::registerOps() {
    // llvm::outs() << "Register " << getDialectNamespace() << " op\n";
    addOperations<
#define GET_OP_LIST
#include "Dialect/SAR/IR/SAROps.cpp.inc"
    >();
}

static LogicalResult verifyBinaryOpShapes(Operation *op) {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto result = op->getResult(0);

    auto lhsType = llvm::dyn_cast<mlir::sar::TensorType>(lhs.getType());
    if (!lhsType)
        return op->emitOpError("expected SAR tensor type for lhs");
    auto rhsType = llvm::dyn_cast<mlir::sar::TensorType>(rhs.getType());
    if (!rhsType)
        return op->emitOpError("expected SAR tensor type for rhs");
    auto resultType = llvm::dyn_cast<mlir::sar::TensorType>(result.getType());
    if (!resultType)
        return op->emitOpError("expected SAR tensor type for result");

    if (lhsType.getShape() != rhsType.getShape() || lhsType.getShape() != resultType.getShape()) {
        return op->emitOpError("operand and result shapes must be equal");
    }
    if (lhsType.getElementType() != rhsType.getElementType() || lhsType.getElementType() != resultType.getElementType()) {
        return op->emitOpError("operand and result element types must be equal");
    }
    return success();
}

static LogicalResult verifyUnaryOpShapes(Operation *op) {
    auto input = op->getOperand(0);
    auto result = op->getResult(0);

    auto inputType = llvm::dyn_cast<mlir::ShapedType>(input.getType());
    if (!inputType)
        return op->emitOpError("expected shaped type for input");
    auto resultType = llvm::dyn_cast<mlir::ShapedType>(result.getType());
    if (!resultType)
        return op->emitOpError("expected shaped type for result");

    if (inputType.getShape() != resultType.getShape()) {
        return op->emitOpError("input and result shapes must be equal");
    }
    if (inputType.getElementType() != resultType.getElementType()) {
        return op->emitOpError("input and result element types must be equal");
    }
    return success();
}

LogicalResult AddOp::verify() {
    return verifyBinaryOpShapes(getOperation());
}

LogicalResult SubOp::verify() {
    return verifyBinaryOpShapes(getOperation());
}

LogicalResult MulOp::verify() {
    return verifyBinaryOpShapes(getOperation());
}

LogicalResult DivOp::verify() {
    return verifyBinaryOpShapes(getOperation());
}

LogicalResult FFT1dOp::verify() {
    return verifyUnaryOpShapes(getOperation());
}

LogicalResult FFT2d0Op::verify() {
    return verifyUnaryOpShapes(getOperation());
}

LogicalResult FFT2d1Op::verify() {
    return verifyUnaryOpShapes(getOperation());
}

} // namespace mlir::sar
