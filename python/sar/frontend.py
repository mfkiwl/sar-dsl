import inspect
from typing import Callable, Tuple

from mlir.ir import Context, InsertionPoint, Location
from mlir import ir as _ir
from mlir.dialects import func

from ._mlir_libs import _sarDialects as _sar_ext
from ._mlir_libs._sarDialects.sar import TensorType, MatrixType, VectorType
from ._sar_ops_gen import (
    FFTnDimOp,
    FFTDimxOp,
    IFFTnDimOp,
    IFFTDimxOp,
    ElemAddOp,
    ElemSubOp,
    ElemMulOp,
    ElemDivOp,
    ConstOp,
    VecMatMulBrdcstOp,
)

def _register_sar(context, load: bool = True) -> None:
    _sar_ext.sar.register_dialect(context, load)


class AutoTypeSpec:
    def __init__(self, dims: Tuple[int, ...] | int, element_type_factory: Callable[[], _ir.Type]):
        if isinstance(dims, int):
            self.dims = (dims,)
        else:
            self.dims = tuple(dims)
        self._factory = element_type_factory

    def materialize(self) -> _ir.Type:
        ctx = Location.current.context
        element_type = self._factory()
        rank = len(self.dims)
        if rank == 1:
            return VectorType.get(list(self.dims), element_type, context=ctx)
        if rank == 2:
            return MatrixType.get(list(self.dims), element_type, context=ctx)
        if rank >= 1:
            return TensorType.get(list(self.dims), element_type, context=ctx)
        raise ValueError("Rank 0 types are not supported for SAR")


class DType:
    def __init__(self, element_type_factory: Callable[[], _ir.Type]):
        self._factory = element_type_factory

    def __getitem__(self, dims: Tuple[int, ...] | int):
        return AutoTypeSpec(dims, self._factory)


int32 = DType(lambda: _ir.IntegerType.get_signless(32))
int64 = DType(lambda: _ir.IntegerType.get_signless(64))
float32 = DType(lambda: _ir.F32Type.get())
float64 = DType(lambda: _ir.F64Type.get())
complex64 = DType(lambda: _ir.ComplexType.get(_ir.F32Type.get()))
complex128 = DType(lambda: _ir.ComplexType.get(_ir.F64Type.get()))


class SARTensor:
    def __init__(self, value: _ir.Value):
        self.value = value

    def __add__(self, other: "SARTensor") -> "SARTensor":
        if not isinstance(other, SARTensor):
            raise TypeError("Can only add SARTensor to SARTensor")
        return SARTensor(ElemAddOp(self.value.type, self.value, other.value).result)

    def __sub__(self, other: "SARTensor") -> "SARTensor":
        if not isinstance(other, SARTensor):
            raise TypeError("Can only subtract SARTensor from SARTensor")
        return SARTensor(ElemSubOp(self.value.type, self.value, other.value).result)

    def __mul__(self, other: "SARTensor") -> "SARTensor":
        if not isinstance(other, SARTensor):
            raise TypeError("Can only multiply SARTensor by SARTensor")
        return SARTensor(ElemMulOp(self.value.type, self.value, other.value).result)

    def __truediv__(self, other: "SARTensor") -> "SARTensor":
        if not isinstance(other, SARTensor):
            raise TypeError("Can only divide SARTensor by SARTensor")
        return SARTensor(ElemDivOp(self.value.type, self.value, other.value).result)


def _as_value(x: "SARTensor | _ir.Value") -> _ir.Value:
    if isinstance(x, SARTensor):
        return x.value
    if isinstance(x, _ir.Value):
        return x
    raise TypeError("Expected SARTensor or mlir.ir.Value")


def _parse_sar_shape_and_elem(ty: _ir.Type) -> tuple[list[int], _ir.Type]:
    st = _ir.ShapedType(ty)
    dyn = _ir.ShapedType.get_dynamic_size()
    dims = [-1 if int(d) == dyn else int(d) for d in st.shape]
    elem_ty = st.element_type
    return dims, elem_ty


def const_like(x: "SARTensor | _ir.Value", scalar: int | float = 0.0) -> SARTensor:
    v = _as_value(x)
    dims, elem_ty = _parse_sar_shape_and_elem(v.type)

    if _ir.ComplexType.isinstance(elem_ty):
        inner = _ir.ComplexType(elem_ty).element_type
        inner_str = "f32" if _ir.F32Type.isinstance(inner) else "f64"
        shape_str = "x".join("?" if d < 0 else str(d) for d in dims)
        real = float(scalar) if isinstance(scalar, (int, float)) else float(scalar.real)
        imag = 0.0 if isinstance(scalar, (int, float)) else float(scalar.imag)
        text = f"dense<({real}, {imag})> : tensor<{shape_str}xcomplex<{inner_str}>>"
        attr = _ir.StringAttr.get(text)
    else:
        std_ty = _ir.RankedTensorType.get(dims, elem_ty)
        if _ir.FloatType.isinstance(elem_ty):
            attr = _ir.DenseElementsAttr.get_splat(
                std_ty, _ir.FloatAttr.get(elem_ty, float(scalar))
            )
        else:
            attr = _ir.DenseElementsAttr.get_splat(
                std_ty, _ir.IntegerAttr.get(elem_ty, int(scalar))
            )

    return SARTensor(ConstOp(v.type, attr).result)


def const(
    shape: tuple[int, ...] | list[int],
    scalar: int | float = 0.0,
    dtype: _ir.Type | None = None,
) -> SARTensor:
    dims = list(shape)
    if any(d is None or int(d) < 0 for d in dims):
        raise ValueError("const requires static shape (no dynamic dimensions).")

    if dtype is not None:
        elem_ty = dtype
        if _ir.ComplexType.isinstance(elem_ty):
            inner = _ir.ComplexType(elem_ty).element_type
            inner_str = "f32" if _ir.F32Type.isinstance(inner) else "f64"
            shape_str = "x".join(str(d) for d in dims)
            real = float(scalar) if isinstance(scalar, (int, float)) else float(scalar.real)
            imag = 0.0 if isinstance(scalar, (int, float)) else float(scalar.imag)
            text = f"dense<({real}, {imag})> : tensor<{shape_str}xcomplex<{inner_str}>>"
            splat = _ir.StringAttr.get(text)
        else:
            ranked = _ir.RankedTensorType.get(dims, elem_ty)
            if _ir.FloatType.isinstance(elem_ty):
                splat = _ir.DenseElementsAttr.get_splat(
                    ranked, _ir.FloatAttr.get(elem_ty, float(scalar))
                )
            else:
                splat = _ir.DenseElementsAttr.get_splat(
                    ranked, _ir.IntegerAttr.get(elem_ty, int(scalar))
                )
    else:
        if isinstance(scalar, complex):
            elem_ty = _ir.ComplexType.get(_ir.F32Type.get())
            shape_str = "x".join(str(d) for d in dims)
            text = f"dense<({float(scalar.real)}, {float(scalar.imag)})> : tensor<{shape_str}xcomplex<f32>>"
            splat = _ir.StringAttr.get(text)
        elif isinstance(scalar, float):
            elem_ty = _ir.F32Type.get()
            splat = _ir.DenseElementsAttr.get_splat(
                _ir.RankedTensorType.get(dims, elem_ty),
                _ir.FloatAttr.get(elem_ty, float(scalar)),
            )
        else:
            elem_ty = _ir.IntegerType.get_signless(32)
            splat = _ir.DenseElementsAttr.get_splat(
                _ir.RankedTensorType.get(dims, elem_ty),
                _ir.IntegerAttr.get(elem_ty, int(scalar)),
            )
    ctx = Location.current.context
    rank = len(dims)
    if rank == 1:
        sar_ty = VectorType.get(dims, elem_ty, context=ctx)
    elif rank == 2:
        sar_ty = MatrixType.get(dims, elem_ty, context=ctx)
    elif rank >= 1:
        sar_ty = TensorType.get(dims, elem_ty, context=ctx)
    else:
        raise ValueError("Rank 0 is unsupported.")

    return SARTensor(ConstOp(sar_ty, splat).result)


def fft_ndim(x: "SARTensor | _ir.Value") -> SARTensor:
    v = _as_value(x)
    return SARTensor(FFTnDimOp(v.type, v).result)


def fft_dimx(x: "SARTensor | _ir.Value", *, dim: int) -> SARTensor:
    v = _as_value(x)
    return SARTensor(FFTDimxOp(v.type, v, dim).result)


def ifft_ndim(x: "SARTensor | _ir.Value") -> SARTensor:
    v = _as_value(x)
    return SARTensor(IFFTnDimOp(v.type, v).result)


def ifft_dimx(x: "SARTensor | _ir.Value", *, dim: int) -> SARTensor:
    v = _as_value(x)
    return SARTensor(IFFTDimxOp(v.type, v, dim).result)


def vec_mat_mul_brdcst(
    vec: "SARTensor | _ir.Value", mat: "SARTensor | _ir.Value", *, dim: int
) -> SARTensor:
    vv = _as_value(vec)
    mm = _as_value(mat)
    return SARTensor(VecMatMulBrdcstOp(mm.type, vv, mm, dim).result)


def _as_mlir_type(t) -> _ir.Type:
    if isinstance(t, _ir.Type):
        return t
    if isinstance(t, AutoTypeSpec):
        return t.materialize()
    raise TypeError(f"Unsupported type annotation: {t}")


def _parse_signature(py_func: Callable) -> Tuple[Tuple[_ir.Type, ...], Tuple[_ir.Type, ...]]:
    sig = inspect.signature(py_func)
    arg_types = []
    for p in sig.parameters.values():
        if p.annotation is inspect.Signature.empty:
            raise TypeError(f"Missing type annotation for parameter '{p.name}'.")
        arg_types.append(_as_mlir_type(p.annotation))

    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError("Missing return type annotation.")

    ret_ann = sig.return_annotation
    res_types = (
        tuple(_as_mlir_type(a) for a in ret_ann)
        if isinstance(ret_ann, tuple)
        else (_as_mlir_type(ret_ann),)
    )
    return tuple(arg_types), res_types


def sar_func(py_func: Callable) -> Callable[[], _ir.Module]:
    def wrapper(*args, **kwargs) -> _ir.Module:
        with Context() as ctx:
            _register_sar(ctx, load=True)
            with Location.name(py_func.__name__):
                module = _ir.Module.create()
                arg_types, res_types = _parse_signature(py_func)

                with InsertionPoint(module.body):
                    # Build FuncOp
                    @func.FuncOp.from_py_func(
                        *arg_types, results=res_types, name=py_func.__name__
                    )
                    def _wrapped(*fun_args):
                        wrapped_args = [SARTensor(arg) for arg in fun_args]
                        ret = py_func(*wrapped_args)
                        if isinstance(ret, SARTensor):
                            func.ReturnOp([ret.value])
                        elif isinstance(ret, tuple) and all(
                            isinstance(v, SARTensor) for v in ret
                        ):
                            func.ReturnOp([v.value for v in ret])
                        else:
                            raise TypeError(
                                "Return value must be SARTensor or tuple of SARTensor."
                            )

                module.operation.verify()
                return module

    return wrapper
