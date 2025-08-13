# python/sar/frontend.py

import inspect
from typing import Callable, Tuple

from mlir.ir import Context, InsertionPoint, Location
from mlir import ir as _ir
from mlir.dialects import func

from . import register_dialect as _register_sar
from ._sar_ops_gen import FFTnDimOp, FFTDimxOp, ElemAddOp
from . import TensorType  # SAR dialect TensorType from C-extension


class TensorTypeSpec:
    """Deferred SAR Tensor type spec materialized within an active Context/Location."""

    def __init__(self, dims: Tuple[int, ...], element_type_factory: Callable[[], _ir.Type]):
        self.dims = tuple(dims)
        self._factory = element_type_factory

    def materialize(self) -> _ir.Type:
        # Use the current context from the current location stack
        ctx = Location.current.context
        element_type = self._factory()
        return TensorType.get(list(self.dims), element_type, context=ctx)


class DType:
    def __init__(self, element_type_factory: Callable[[], _ir.Type]):
        self._factory = element_type_factory

    def __getitem__(self, dims: Tuple[int, ...]):
        # Defer materialization until we are inside a Context/Location
        return TensorTypeSpec(dims, self._factory)


float32 = DType(lambda: _ir.F32Type.get())


class SARTensor:
    """Thin wrapper around mlir.ir.Value for SAR frontend operator overloading."""

    def __init__(self, value: _ir.Value):
        self.value = value

    def __add__(self, other: "SARTensor") -> "SARTensor":
        if not isinstance(other, SARTensor):
            raise TypeError("Can only add SARTensor to SARTensor")
        # ElemAddOp(result_type, lhs, rhs)
        return SARTensor(ElemAddOp(self.value.type, self.value, other.value).result)


def _as_value(x: "SARTensor | _ir.Value") -> _ir.Value:
    if isinstance(x, SARTensor):
        return x.value
    if isinstance(x, _ir.Value):
        return x
    raise TypeError("Expected SARTensor or mlir.ir.Value")


def fft_ndim(x: "SARTensor | _ir.Value") -> SARTensor:
    """N-dim FFT on a SAR shaped value."""
    v = _as_value(x)
    return SARTensor(FFTnDimOp(v.type, v).result)


def fft_dimx(x: "SARTensor | _ir.Value", *, dim: int) -> SARTensor:
    """1D FFT along a given dimension (0..2)."""
    v = _as_value(x)
    # Pass plain int; binding will build I64Attr as needed.
    return SARTensor(FFTDimxOp(v.type, v, dim).result)


def _as_mlir_type(t) -> _ir.Type:
    if isinstance(t, _ir.Type):
        return t
    if isinstance(t, TensorTypeSpec):
        return t.materialize()
    raise TypeError(f"Unsupported type annotation: {t}")


def _parse_signature(py_func: Callable) -> Tuple[Tuple[_ir.Type, ...], Tuple[_ir.Type, ...]]:
    sig = inspect.signature(py_func)
    # Positional only for simplicity; return annotation in 'return'.
    arg_types = []
    for p in sig.parameters.values():
        if p.annotation is inspect._empty:
            raise TypeError(f"Missing type annotation for parameter {p.name}")
        arg_types.append(_as_mlir_type(p.annotation))
    if sig.return_annotation is inspect._empty:
        raise TypeError("Missing return type annotation")
    ret_ann = sig.return_annotation
    if isinstance(ret_ann, tuple):
        res_types = tuple(_as_mlir_type(a) for a in ret_ann)
    else:
        res_types = (_as_mlir_type(ret_ann),)
    return tuple(arg_types), res_types


def sar_func(py_func: Callable) -> Callable[[], _ir.Module]:
    """Decorator: define a SAR func.func from a typed Python function.

    The function annotations must use SAR dtype helpers, e.g. float32[8,4,2].
    """

    def wrapper(*args, **kwargs):
        with Context() as ctx:
            _register_sar(ctx, load=True)
            with Location.unknown():
                module = _ir.Module.create()
                arg_types, res_types = _parse_signature(py_func)
                with InsertionPoint(module.body):
                    @func.FuncOp.from_py_func(*arg_types, results=res_types, name=py_func.__name__)
                    def _wrapped(*fun_args):
                        wrapped_args = [SARTensor(arg) for arg in fun_args]
                        ret = py_func(*wrapped_args)
                        if isinstance(ret, SARTensor):
                            func.ReturnOp([ret.value])
                        elif isinstance(ret, tuple) and all(isinstance(v, SARTensor) for v in ret):
                            func.ReturnOp([v.value for v in ret])
                        else:
                            raise TypeError("Return value must be SARTensor or tuple of SARTensor")
                return module

    return wrapper 