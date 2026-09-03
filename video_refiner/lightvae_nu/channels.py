"""Channel schedules for the causal non-uniform Wan2.2 decoder student.

Teacher Wan2.2 decoder (dec_dim=256, dim_mult=[1,2,4,4]):
    dims = [1024, 1024, 1024, 512, 256]
    # conv1/mid, up0 (T×2,HW×2), up1 (T×2,HW×2), up2 (HW×2), up3+head (no up)

scheme1 keeps the last spatial stage nearly full-width (PSNR lives there)
and prunes the low-resolution 3D convs (where FLOPs live).
"""
from __future__ import annotations

from typing import Sequence, Tuple

# Teacher reference (must match wan.modules.vae2_2.Decoder3d with dec_dim=256).
TEACHER_DIMS: Tuple[int, ...] = (1024, 1024, 1024, 512, 256)

# scheme1: mid/up0/up1 keep 0.25, up2 keep 0.75, head keep 1.0
SCHEME1_DIMS: Tuple[int, ...] = (256, 256, 256, 384, 256)

# scheme2: high-res stages slimmed for speed (measured 2K H20 bf16):
# 157 ms/pix-frame vs scheme1 383 (~2.4x), decoder 45M.
# up2 keep 1.0 (256), up3+head keep 0.5 (128).
SCHEME2_DIMS: Tuple[int, ...] = (256, 256, 256, 256, 128)

# scheme3: high-res budget equal to LightVAE-v2 (measured 76 ms/pix-frame,
# ~same as v2's 72); low-res stages stay at v2 width too. Speed floor.
SCHEME3_DIMS: Tuple[int, ...] = (256, 256, 256, 128, 64)

# CPU tests. Same keep-ratio shape, 16× smaller.
TINY_DIMS: Tuple[int, ...] = (16, 16, 16, 24, 16)

# Wan2.2_VAE wrapper uses temperal_downsample=[False, True, True], so decoder
# temperal_upsample is the reverse: [True, True, False] for the first 3 up
# blocks (the 4th has up_flag=False and ignores the flag).
TEMPORAL_UPSAMPLE: Tuple[bool, ...] = (True, True, False)

Z_DIM = 48
PATCH_SIZE = 2
SPATIAL_COMPRESSION = 16
TEMPORAL_COMPRESSION = 4


def pixel_frames(latent_frames: int) -> int:
    """Wan2.2 first-frame special case: T_pix = 4*(T_lat-1)+1."""
    if latent_frames < 1:
        raise ValueError(f"latent_frames must be >= 1, got {latent_frames}")
    return 4 * (latent_frames - 1) + 1


def latent_frames_for_pixels(pixel_n: int) -> int:
    """Invert pixel_frames. pixel_n must be 4k+1."""
    if pixel_n < 1 or (pixel_n - 1) % 4 != 0:
        raise ValueError(f"pixel frame count must be 4k+1, got {pixel_n}")
    return (pixel_n - 1) // 4 + 1


def validate_dims(dims: Sequence[int]) -> Tuple[int, ...]:
    """Assert DupUp3D divisibility for the Wan2.2 upsample pattern.

    Up blocks i=0,1: temporal+spatial, factor = 2*2*2 = 8
    Up block i=2: spatial only, factor = 1*2*2 = 4
    Up block i=3: no DupUp3D
    """
    dims = tuple(int(d) for d in dims)
    if len(dims) != 5:
        raise ValueError(f"expected 5 stage dims (conv1 + 4 up), got {dims}")
    if any(d <= 0 for d in dims):
        raise ValueError(f"dims must be positive, got {dims}")
    # i=0,1: in=dims[i], out=dims[i+1], factor=8
    for i, factor in ((0, 8), (1, 8), (2, 4)):
        in_c, out_c = dims[i], dims[i + 1]
        if (out_c * factor) % in_c != 0:
            raise ValueError(
                f"DupUp3D failed at up[{i}]: out {out_c} * factor {factor} "
                f"not divisible by in {in_c}"
            )
    return dims


NAMED_SCHEDULES = {
    "scheme1": SCHEME1_DIMS,
    "scheme2": SCHEME2_DIMS,
    "scheme3": SCHEME3_DIMS,
    "tiny": TINY_DIMS,
    "teacher": TEACHER_DIMS,
}
