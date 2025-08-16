from mlir.dialects.sar.frontend import *

@sar_func
def forward(
    a: float32[8, 4, 2],      # f32 3D
    b: float32[8, 4, 2],      # f32 3D (same as a)
    c: float32[16, 2, 1],     # f32 3D (different shape)
    x: int32[5, 5, 1],        # i32 3D
    y: int32[5, 5, 1],        # i32 3D (same as x)
    m1: float32[4, 4],        # f32 2D Matrix
    m2: float32[4, 4],        # f32 2D Matrix
    v1: float32[4],           # f32 1D Vector
    v2: float32[4],           # f32 1D Vector
    t: complex64[4, 4],       # c64 2D Matrix
) -> (
    float32[8, 4, 2],         # z
    float32[8, 4, 2],         # u
    float32[16, 2, 1],        # v
    float32[16, 2, 1],        # v_ifft
    int32[5, 5, 1],           # w
    float32[4, 4],            # mm
    float32[4],               # vv
    float32[4, 4],            # mm2 (vec-mat brdcst)
    complex64[4, 4],          # cm (complex)
):
    # Float branch (3D f32)
    ca0 = const_like(a)
    ca1 = const_like(a, 1.0)
    ca2 = const([8, 4, 2], 2.0)

    fa = fft_ndim(a)
    fb = fft_dimx(b, dim=1)

    z = (fa + ca1) * (fb + ca2) - (a / (b + ca1))
    u = (a + b) - ca0

    # Another float shape with ifft
    cc0 = const_like(c)
    cc3 = const([16, 2, 1], 3.0)
    v = fft_dimx((c + cc3) - cc0, dim=0)
    v_ifft = ifft_dimx(ifft_ndim(v), dim=0)

    # Int branch (3D i32)
    cx0 = const_like(x)
    cx5 = const([5, 5, 1], 5)
    w = (x + y) * (y - cx5) - cx0

    # Matrix/Vector branch
    mm = (m1 + m2) - const_like(m1)
    vv = v1 + v2

    # vec-mat broadcast: vector length must match matrix dim along 'dim'
    mm2 = vec_mat_mul_brdcst(v1, m1, dim=0)

    # Complex branch (2D c64)
    cm = t + const_like(t, 1.0 + 2.0j)

    return z, u, v, v_ifft, w, mm, vv, mm2, cm

print("=" * 50)
print("SAR Dialect Compr Test")
print("=" * 50)

module = forward()
print(module)
