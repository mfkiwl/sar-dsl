#include "mlir/IR/Types.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/AsmParser/AsmParser.h"

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

    auto lhsType = llvm::dyn_cast<mlir::ShapedType>(lhs.getType());
    if (!lhsType)
        return op->emitOpError("expected shaped type for lhs");
    auto rhsType = llvm::dyn_cast<mlir::ShapedType>(rhs.getType());
    if (!rhsType)
        return op->emitOpError("expected shaped type for rhs");
    auto resultType = llvm::dyn_cast<mlir::ShapedType>(result.getType());
    if (!resultType)
        return op->emitOpError("expected shaped type for result");

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

LogicalResult ConstOp::verify() {
    auto resTy = llvm::dyn_cast<mlir::ShapedType>(getResult().getType());
    if (!resTy)
        return emitOpError("result must be a shaped SAR type");

    Attribute raw = getValueAttr();
    DenseElementsAttr elems;
    if (auto de = llvm::dyn_cast<DenseElementsAttr>(raw)) {
        elems = de;
    } else if (auto s = llvm::dyn_cast<StringAttr>(raw)) {
        Attribute parsed = mlir::parseAttribute(s.getValue(), getOperation()->getContext());
        elems = llvm::dyn_cast_or_null<DenseElementsAttr>(parsed);
        if (!elems)
            return emitOpError("value StringAttr did not parse to DenseElementsAttr");
    } else {
        return emitOpError("value must be DenseElementsAttr or StringAttr");
    }

    auto ranked = llvm::dyn_cast<RankedTensorType>(elems.getType());
    if (!ranked)
        return emitOpError("dense elements must have ranked tensor type");

    if (ranked.getElementType() != resTy.getElementType())
        return emitOpError("element type mismatch between value and result type");

    if (ranked.getShape() != resTy.getShape())
        return emitOpError("shape mismatch between value and result type");

    return success();
}

LogicalResult ElemAddOp::verify() {
    return verifyBinaryOpShapes(getOperation());
}

LogicalResult ElemSubOp::verify() {
    return verifyBinaryOpShapes(getOperation());
}

LogicalResult ElemMulOp::verify() {
    return verifyBinaryOpShapes(getOperation());
}

LogicalResult ElemDivOp::verify() {
    return verifyBinaryOpShapes(getOperation());
}

LogicalResult FFTnDimOp::verify() {
    return verifyUnaryOpShapes(getOperation());
}

LogicalResult IFFTnDimOp::verify() {
    return verifyUnaryOpShapes(getOperation());
}

LogicalResult FFTDimxOp::verify() {
    if (failed(verifyUnaryOpShapes(getOperation())))
        return failure();
    auto inputType = llvm::dyn_cast<mlir::ShapedType>(getInput().getType());
    if (!inputType)
        return emitOpError("expected shaped type for input");
    auto dimAttr = getDimAttr();
    if (!dimAttr)
        return emitOpError("missing dim attribute");
    int64_t d = dimAttr.getInt();
    if (d < 0)
        return emitOpError("dim must be non-negative");
    if (d >= inputType.getRank())
        return emitOpError("dim must be less than input rank");
    return success();
}

LogicalResult IFFTDimxOp::verify() {
    if (failed(verifyUnaryOpShapes(getOperation())))
        return failure();
    auto inputType = llvm::dyn_cast<mlir::ShapedType>(getInput().getType());
    if (!inputType)
        return emitOpError("expected shaped type for input");
    auto dimAttr = getDimAttr();
    if (!dimAttr)
        return emitOpError("missing dim attribute");
    int64_t d = dimAttr.getInt();
    if (d < 0)
        return emitOpError("dim must be non-negative");
    if (d >= inputType.getRank())
        return emitOpError("dim must be less than input rank");
    return success();
}

LogicalResult VecMatMulBrdcstOp::verify() {
    auto vecType = llvm::dyn_cast<mlir::sar::VectorType>(getLhs().getType());
    if (!vecType)
        return emitOpError("expected SAR vector type for lhs");
    auto matType = llvm::dyn_cast<mlir::sar::MatrixType>(getRhs().getType());
    if (!matType)
        return emitOpError("expected SAR matrix type for rhs");
    auto resType = llvm::dyn_cast<mlir::sar::MatrixType>(getResult().getType());
    if (!resType)
        return emitOpError("expected SAR matrix type for result");

    auto dimAttr = getDimAttr();
    if (!dimAttr)
        return emitOpError("missing dim attribute");
    int64_t d = dimAttr.getInt();
    if (d != 0 && d != 1)
        return emitOpError("dim must be 0 or 1");

    auto matShape = matType.getShape();
    auto resShape = resType.getShape();
    if (matShape != resShape)
        return emitOpError("result shape must equal matrix shape");

    int64_t vecLen = vecType.getShape().front();
    if (vecLen < 0)
        return emitOpError("vector length must be static for verification");
    if (matShape[d] != vecLen)
        return emitOpError("vector length must match matrix dimension along 'dim'");

    if (vecType.getElementType() != matType.getElementType() ||
        vecType.getElementType() != resType.getElementType())
        return emitOpError("element types of vec, mat, result must be equal");

    return success();
}

} // namespace mlir::sar
