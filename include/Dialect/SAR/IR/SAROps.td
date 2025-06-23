// include/Dialect/SAR/IR/SAROps.td

#ifndef DIALECT_SAR_OPS_TD
#define DIALECT_SAR_OPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "Dialect/SAR/IR/SARDialect.td"
include "Dialect/SAR/IR/SARTypes.td"

// Base class for SAR-specific operations
class SAR_Op<string mnemonic, list<Trait> traits = []>
    : Op<SAR_Dialect, mnemonic, traits> {
}

// Base class for binary operations
class SAR_BinaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [Pure], dag attributes = (ins)>
    : SAR_Op<mnemonic, traits#[]> {
    let arguments = !con((ins
        OperandType:$lhs,
        OperandType:$rhs),
        attributes
    );
    let results = (outs
        resultType:$result);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            build($_builder, $_state, lhs.getType(), lhs, rhs);
        }]>
    ];
}

// Constant value operation
def ConstOp : SAR_Op<"const", [Pure, ConstantLike]> {
    let arguments = (ins
        ElementsAttr:$value);
    let results = (outs
        AnySARTensor:$result);
}

// Arithmetic operations
def AddOp : SAR_BinaryOp<"add", AnySARTensor>;
def SubOp : SAR_BinaryOp<"sub", AnySARTensor>;
def MulOp : SAR_BinaryOp<"mul", AnySARTensor>;
def DivOp : SAR_BinaryOp<"div", AnySARTensor>;

// Print operation
def PrintOp: SAR_Op<"print", []> {
    let arguments = (ins
        AnyType:$input
    );
}

#endif
