import sys as _sys
import ctypes as _ctypes
import os as _os


def _preload_upstream_mlir_and_capi() -> None:
    _old_flags = None
    try:
        if hasattr(_sys, "setdlopenflags") and hasattr(_ctypes, "RTLD_GLOBAL"):
            _old_flags = _sys.getdlopenflags()
            _sys.setdlopenflags(_old_flags | _ctypes.RTLD_GLOBAL)
        from mlir._mlir_libs import _mlir as _mlir_cext
    finally:
        if _old_flags is not None:
            _sys.setdlopenflags(_old_flags)

    try:
        from mlir._mlir_libs import get_lib_dirs as _mlir_get_lib_dirs
        capi_loaded = False
        for _d in _mlir_get_lib_dirs():
            try:
                for _name in _os.listdir(_d):
                    if _name.startswith("libMLIRPythonCAPI.so"):
                        _ctypes.CDLL(_os.path.join(_d, _name), mode=_ctypes.RTLD_GLOBAL)
                        capi_loaded = True
                        break
                if capi_loaded:
                    break
            except FileNotFoundError:
                continue
    except Exception:
        pass


_preload_upstream_mlir_and_capi()
from mlir.ir import *
from ._sar_ops_gen import *
from ._mlir_libs import _sarDialects as _sar_ext
from ._mlir_libs._sarDialects.sar import TensorType, MatrixType, VectorType

from .frontend import (
    sar_func,
    float32,
    int32,
    float64,
    int64,
    complex64,
    complex128,
    fft_ndim,
    fft_dimx,
    ifft_ndim,
    ifft_dimx,
    vec_mat_mul_brdcst,
)


def lower_to_linalg_text(ir_text: str) -> str:
    return _sar_ext.sar.lower_to_linalg(ir_text)


def register_dialect(context=None, load: bool = True) -> None:
    _sar_ext.sar.register_dialect(context, load)
