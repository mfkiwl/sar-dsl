// include/Dialect/SAR/IR/SARConstraints.td

#ifndef DIALECT_SAR_CONSTRAINTS_TD
#define DIALECT_SAR_CONSTRAINTS_TD

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def AnySARSharedType : Type<Or<[CPred<"::mlir::isa<::mlir::sar::TensorType>($_self)">,
                                CPred<"::mlir::isa<::mlir::sar::MatrixType>($_self)">,
                                CPred<"::mlir::isa<::mlir::sar::VectorType>($_self)">]>>;

def AnySARTensor : Type<And<[CPred<"::mlir::isa<::mlir::sar::TensorType>($_self)">,
                             CPred<"::mlir::cast<::mlir::sar::TensorType>($_self).getShape().size() == 3">]>>;

def AnySARMatrix : Type<And<[CPred<"::mlir::isa<::mlir::sar::MatrixType>($_self)">,
                             CPred<"::mlir::cast<::mlir::sar::MatrixType>($_self).getShape().size() == 2">]>>;

def AnySARVector : Type<And<[CPred<"::mlir::isa<::mlir::sar::VectorType>($_self)">,
                             CPred<"::mlir::cast<::mlir::sar::VectorType>($_self).getShape().size() == 1">]>>;

#endif
