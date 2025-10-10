from mlir.dialects.sar.frontend import *

@sar_func
def sar_range_compression(
    echo: complex64[128, 256],
    chirp: complex64[128, 256],
) -> complex64[128, 256]:
    echo_fft = fft_dimx(echo, dim=0)
    chirp_fft = fft_dimx(chirp, dim=0)
    compressed_fft = echo_fft * chirp_fft
    compressed = ifft_dimx(compressed_fft, dim=0)
    return compressed

@sar_func
def sar_azimuth_compression(
    range_compressed: complex64[128, 256],
    azimuth_ref: complex64[128, 256],
) -> complex64[128, 256]:
    rc_fft = fft_dimx(range_compressed, dim=1)
    az_ref_fft = fft_dimx(azimuth_ref, dim=1)
    focused_fft = rc_fft * az_ref_fft
    sar_image = ifft_dimx(focused_fft, dim=1)
    return sar_image

@sar_func
def matched_filter(
    signal: complex64[128, 256],
    reference: complex64[128, 256],
) -> complex64[128, 256]:
    signal_fft = fft_ndim(signal)
    reference_fft = fft_ndim(reference)
    matched_freq = signal_fft * reference_fft
    matched_result = ifft_ndim(matched_freq)
    return matched_result

print("=" * 50)
print("SAR Range Compression")
print("=" * 50)

module1 = sar_range_compression()
print(module1)

print("=" * 50)
print("SAR Azimuth Compression")
print("=" * 50)

module2 = sar_azimuth_compression()
print(module2)

print("=" * 50)
print("Matched Filter")
print("=" * 50)

module3 = matched_filter()
print(module3)
