#include "mlir/IR/Types.h"
#include "mlir/CAPI/Registration.h"
#include "mlir-c/Support.h"
#include "llvm/Support/Casting.h"
#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"
#include "SAR-c/Dialects.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SAR, sar, mlir::sar::SARDialect)

MlirType mlirSARTensorTypeGet(MlirContext ctx, intptr_t numDims, const int64_t *dims, MlirType elementType) {
    return wrap(mlir::sar::TensorType::get(unwrap(ctx), llvm::ArrayRef<int64_t>(dims, numDims), unwrap(elementType)));
}

bool mlirTypeIsASARTensorType(MlirType t) {
    return llvm::isa<mlir::sar::TensorType>(unwrap(t));
}

MlirTypeID mlirSARTensorTypeGetTypeID(void) {
    return wrap(mlir::sar::TensorType::getTypeID());
}

MlirType mlirSARMatrixTypeGet(MlirContext ctx, intptr_t numDims, const int64_t *dims, MlirType elementType) {
    return wrap(mlir::sar::MatrixType::get(unwrap(ctx), llvm::ArrayRef<int64_t>(dims, numDims), unwrap(elementType)));
}

bool mlirTypeIsASARMatrixType(MlirType t) {
    return llvm::isa<mlir::sar::MatrixType>(unwrap(t));
}

MlirTypeID mlirSARMatrixTypeGetTypeID(void) {
    return wrap(mlir::sar::MatrixType::getTypeID());
}

MlirType mlirSARVectorTypeGet(MlirContext ctx, intptr_t numDims, const int64_t *dims, MlirType elementType) {
    return wrap(mlir::sar::VectorType::get(unwrap(ctx), llvm::ArrayRef<int64_t>(dims, numDims), unwrap(elementType)));
}

bool mlirTypeIsASARVectorType(MlirType t) {
    return llvm::isa<mlir::sar::VectorType>(unwrap(t));
}

MlirTypeID mlirSARVectorTypeGetTypeID(void) {
    return wrap(mlir::sar::VectorType::getTypeID());
}
