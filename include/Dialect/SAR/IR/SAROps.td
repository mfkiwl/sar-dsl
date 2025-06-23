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

class SAR_Op<string mnemonic, list<Trait> traits = []>
    : Op<SAR_Dialect, mnemonic, traits> {
}

class SAR_BinaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [Pure], dag attributes = (ins)>
    : SAR_Op<mnemonic, traits#[]> {
    let arguments = !con((ins
        OperandType:$lhs,
        OperandType:$rhs),
        attributes
    );
    let results = (outs
        resultType:$result);
}

def ConstOp : SAR_Op<"const", [Pure, ConstantLike]> {
    let arguments = (ins
        ElementsAttr:$value);
    let results = (outs
        AnySARTensor:$result);
}

def AddOp : SAR_BinaryOp<"add", AnySARTensor>;
def SubOp : SAR_BinaryOp<"sub", AnySARTensor>;
def MulOp : SAR_BinaryOp<"mul", AnySARTensor>;
def DivOp : SAR_BinaryOp<"div", AnySARTensor>;

def PrintOp: SAR_Op<"print", []> {
    let arguments = (ins
        AnyType:$input
    );
}

#endif
