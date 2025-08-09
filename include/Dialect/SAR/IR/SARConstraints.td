// include/Dialect/SAR/IR/SARConstraints.td

#ifndef DIALECT_SAR_CONSTRAINTS_TD
#define DIALECT_SAR_CONSTRAINTS_TD

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def AnySARTensor : Type<And<[CPred<"::mlir::isa<::mlir::sar::TensorType, ::mlir::sar::MatrixType, ::mlir::sar::VectorType>($_self)">,
                             CPred<"::mlir::cast<::mlir::ShapedType>($_self).getShape().size() > 0">]>>;

def AnySARVector : Type<And<[CPred<"::mlir::isa<::mlir::sar::VectorType>($_self)">,
                             CPred<"::mlir::cast<::mlir::sar::VectorType>($_self).getShape().size() == 1">]>>;

def AnySARMatrix : Type<And<[CPred<"::mlir::isa<::mlir::sar::MatrixType>($_self)">,
                             CPred<"::mlir::cast<::mlir::sar::MatrixType>($_self).getShape().size() == 2">]>>;



#endif
