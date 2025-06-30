// include/Dialect/SAR/IR/SARConstraints.td

#ifndef DIALECT_SAR_CONSTRAINTS_TD
#define DIALECT_SAR_CONSTRAINTS_TD

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def AnySARTensor : Type<And<[CPred<"::mlir::isa<::mlir::sar::TensorType>($_self)">,
                             CPred<"::mlir::cast<::mlir::sar::TensorType>($_self).getShape().size() > 0">]>>;

#endif
