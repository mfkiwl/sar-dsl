#ifndef DIALECT_SAR_OPS_H
#define DIALECT_SAR_OPS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "Dialect/SAR/IR/SARDialect.h"
#include "Dialect/SAR/IR/SARTypes.h"

#define GET_OP_CLASSES
#include "Dialect/SAR/IR/SAROps.h.inc"

#endif
