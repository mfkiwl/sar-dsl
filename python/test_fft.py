# python/test_fft.py

from mlir.dialects.sar import sar_func, float32, fft_ndim, fft_dimx

@sar_func
def forward(arg0: float32[8, 4, 2], arg1: float32[8, 4, 2]) -> float32[8, 4, 2]:
    var0 = fft_ndim(arg0)
    var1 = fft_dimx(arg1, dim=1)
    return var0 + var1

module = forward()
print(module)
