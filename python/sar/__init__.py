# python/sar/__init__.py

import sys as _sys
import ctypes as _ctypes
import os as _os


def _preload_upstream_mlir_and_capi() -> None:
    # Load upstream _mlir with RTLD_GLOBAL if possible.
    _old_flags = None
    try:
        if hasattr(_sys, "setdlopenflags") and hasattr(_ctypes, "RTLD_GLOBAL"):
            _old_flags = _sys.getdlopenflags()
            _sys.setdlopenflags(_old_flags | _ctypes.RTLD_GLOBAL)
        from mlir._mlir_libs import _mlir as _mlir_cext  # noqa: F401
    finally:
        # Always restore flags if we changed them.
        if _old_flags is not None:
            _sys.setdlopenflags(_old_flags)

    # Proactively load libMLIRPythonCAPI with RTLD_GLOBAL if available.
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

from mlir.ir import *  # noqa: F401,F403
from ._sar_ops_gen import *  # noqa: F401,F403
from ._sar_ops_gen import _Dialect

# Load the native extension module and expose a registration helper.
from ._mlir_libs import _sarDialects as _sar_ext
from ._mlir_libs._sarDialects.sar import TensorType, MatrixType, VectorType


def register_dialect(context=None, load: bool = True) -> None:
    """Register (and optionally load) the SAR dialect into a given MLIR context.

    Args:
        context: mlir.ir.Context or None (defaults to current/implicit context usage).
        load: Whether to also load the dialect into the context.
    """
    _sar_ext.sar.register_dialect(context, load)

from .frontend import sar_func, float32, fft_ndim, fft_dimx  # noqa: F401
