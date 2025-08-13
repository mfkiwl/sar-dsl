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

class SAR_BinaryOp<string mnemonic,
                   Type lhsType,
                   Type rhsType,
                   Type resultType,
                   list<Trait> traits = [Pure],
                   dag attributes = (ins)>
    : SAR_Op<mnemonic, traits # []> {
    let arguments = !con((ins
        lhsType:$lhs,
        rhsType:$rhs),
        attributes);
    let results = (outs
        resultType:$result);
}

class SAR_UnaryOp<string mnemonic,
                   Type operandType,
                   Type resultType,
                   list<Trait> traits = [Pure],
                   dag attributes = (ins)>
    : SAR_Op<mnemonic, traits # []> {
    let arguments = !con((ins
        operandType:$input),
        attributes);
    let results = (outs
        resultType:$result);
}

def ConstOp : SAR_Op<"const", [Pure]> {
    let arguments = (ins ElementsAttr:$value);
    let results = (outs AnySARSharedType:$result);
}

def ElemAddOp : SAR_BinaryOp<"elem_add", AnySARSharedType, AnySARSharedType, AnySARSharedType> { let hasVerifier = 1; }
def ElemSubOp : SAR_BinaryOp<"elem_sub", AnySARSharedType, AnySARSharedType, AnySARSharedType> { let hasVerifier = 1; }
def ElemMulOp : SAR_BinaryOp<"elem_mul", AnySARSharedType, AnySARSharedType, AnySARSharedType> { let hasVerifier = 1; }
def ElemDivOp : SAR_BinaryOp<"elem_div", AnySARSharedType, AnySARSharedType, AnySARSharedType> { let hasVerifier = 1; }

def FFTnDimOp  : SAR_UnaryOp<"fft_ndim",  AnySARSharedType, AnySARSharedType> { let hasVerifier = 1; }
def IFFTnDimOp : SAR_UnaryOp<"ifft_ndim", AnySARSharedType, AnySARSharedType> { let hasVerifier = 1; }

def FFTDimxOp  : SAR_UnaryOp<"fft_dimx",  AnySARSharedType, AnySARSharedType, [Pure], (ins I64Attr:$dim)> { let hasVerifier = 1; }
def IFFTDimxOp : SAR_UnaryOp<"ifft_dimx", AnySARSharedType, AnySARSharedType, [Pure], (ins I64Attr:$dim)> { let hasVerifier = 1; }

def VecMatMulBrdcstOp
    : SAR_BinaryOp<"vec_mat_mul_brdcst",
                    AnySARVector, AnySARMatrix, AnySARMatrix,
                    [Pure], (ins I64Attr:$dim)> {
    let hasVerifier = 1;
}

#endif
