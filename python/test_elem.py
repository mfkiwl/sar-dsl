# python/test_elem.py

from mlir import ir as _ir
from mlir.dialects.sar import sar_func, float32, ElemAddOp, ElemMulOp, ElemSubOp, ConstOp
from mlir.dialects.sar.frontend import SARTensor

@sar_func
def forward(arg0: float32[3, 3, 3], arg1: float32[3, 3, 3]) -> float32[3, 3, 3]:
    tensor_type = arg0.value.type
    f32 = _ir.F32Type.get()
    std_tensor_ty = _ir.RankedTensorType.get([3, 3, 3], f32)
    one = _ir.DenseElementsAttr.get_splat(std_tensor_ty, _ir.FloatAttr.get(f32, 1.0))
    two = _ir.DenseElementsAttr.get_splat(std_tensor_ty, _ir.FloatAttr.get(f32, 2.0))

    const1 = ConstOp(tensor_type, one).result
    const2 = ConstOp(tensor_type, two).result

    add = ElemAddOp(tensor_type, arg0.value, const1).result
    mul = ElemMulOp(tensor_type, arg1.value, const2).result
    sub = ElemSubOp(tensor_type, add, mul).result
    return SARTensor(sub)

module = forward()
print(module)
