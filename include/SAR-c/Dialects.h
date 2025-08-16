#ifndef SAR_C_DIALECTS_H
#define SAR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SAR, sar);

MLIR_CAPI_EXPORTED MlirType mlirSARTensorTypeGet(MlirContext ctx, intptr_t numDims, const int64_t *dims, MlirType elementType);
MLIR_CAPI_EXPORTED bool mlirTypeIsASARTensorType(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSARTensorTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirSARMatrixTypeGet(MlirContext ctx, intptr_t numDims, const int64_t *dims, MlirType elementType);
MLIR_CAPI_EXPORTED bool mlirTypeIsASARMatrixType(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSARMatrixTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirSARVectorTypeGet(MlirContext ctx, intptr_t numDims, const int64_t *dims, MlirType elementType);
MLIR_CAPI_EXPORTED bool mlirTypeIsASARVectorType(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSARVectorTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // SAR_C_DIALECTS_H
