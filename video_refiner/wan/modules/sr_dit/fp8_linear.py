"""fp8 (e4m3) row-wise dynamic-scaled Linear for the DiT GEMMs.

WHY fp8 AND NOT A BETTER bf16 SCHEDULE
--------------------------------------
On a 5B SR-DiT block (48x72 token grid, dim 3072, ffn 14336, 30 blocks), one
denoise block measures ~46 ms, split:

    gemm  21.5 ms (47.1%)   <- ~140 TFLOPS = 95% of the bf16 roofline on sm90
    attn  15.9 ms (34.8%)   <- already the fused block-grid Triton kernel
    cast   4.4 ms ( 9.6%)   <- 44 kernels, the fp32 modulation chain
    elem   3.2 ms ( 6.9%)
    norm   0.8 ms ( 1.7%)

The GEMMs are AT the bf16 roofline, so there is no scheduling, layout or
fusion change that can speed them up. The only remaining lever is a
lower-precision datapath: sm90 does 296 TFLOPS of fp8, twice its bf16 rate.

MEASURED, at the real ffn shape (M=10368, K=3072, N=14336):

    bf16 mm          6.531 ms   139.8 TFLOPS
    fp8 rowwise      3.769 ms   242.3 TFLOPS   1.73x
    fp8 tensorwise   3.281 ms   278.3 TFLOPS   1.99x   <- NOT used, see below

Per block that is 21.5 -> 12.4 ms of GEMM (saves 9.1 ms) against 0.8 ms of
activation quantisation, so ~8.3 ms/block net = ~250 ms/forward. Confirmed on
the real block end-to-end: 46.065 -> 37.494 ms (1.23x), gemm 21.551 ->
11.977 ms, with +0.98 ms/block of quantisation -- the bias was folded into
the _scaled_mm epilogue (adding it as a separate op instead cost
+1.60 ms/block).

WHY ROW-WISE AND NOT PER-TENSOR
-------------------------------
Per-tensor scaling is another 15% faster but a single scale for the whole
activation matrix is set by the largest outlier in it, so every other row
loses mantissa bits. Row-wise (per token for the activation, per output
channel for the weight) costs one extra reduction and keeps each row's own
dynamic range -- the standard recipe, and the accuracy difference on real DiT
activations is far larger than the 15%.

fp8 is NOT numerically free: at the same GEMM, rel err is ~3.7e-2 against an
fp32 reference where bf16 is ~1.7e-3 -- about 22x worse, compounding over 30
blocks. That is a quality decision, not a correctness one, which is why this
whole path is OPT-IN (`--fp8_linear`) and why `skip_first`/
`skip_last`/`targets` exist to trade speed back for accuracy.

WHAT IS DELIBERATELY NOT DONE
-----------------------------
* q/k/v share one input (`self.q(x)`, `self.k(x)`, `self.v(x)`), so quantising
  once for all three would drop 2 of the 10 quantisers. Measured: the 10 cost
  0.98 ms/block together, so this saves 0.2 ms of a 37.5 ms block (0.5%). Not
  worth reaching into WanAttentionBlock for: keeping this a pure module swap
  means models.py is untouched and training cannot regress.
* A hand-written Triton quantiser would beat torch.compile's ~1.6 TB/s, but
  only by ~0.17 s over a 45-frame clip (1%). torch.compile(dynamic=False) is
  used instead, with `prewarm_quantizer` to keep inductor's per-shape compile
  off the inference path -- the same lesson the block-grid kernel taught.
"""
import os

import torch
import torch.nn as nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)      # 448.0

# Suffixes of the Linears worth converting: every GEMM inside a transformer
# block. Everything else (patch_embedding, time/text embeddings, Head) is tiny
# and sits where a rounding error is not averaged over 30 layers.
TARGETS_ALL = (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
)
# 60% of the block's GEMM FLOPs live in the FFN. Converting only those keeps
# the attention projections exact -- the accuracy/speed dial if full fp8 is
# too lossy on a given checkpoint.
TARGETS_FFN = ("ffn.0", "ffn.2")


def fp8_supported(device=None):
    """True if this device has fp8 tensor cores and torch exposes _scaled_mm."""
    if not hasattr(torch, "_scaled_mm"):
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(
        device if device is not None else None)
    return (major, minor) >= (8, 9)


def _quantize_rowwise_impl(x):
    # amax in fp32: a bf16 amax of a large row saturates its 8-bit exponent far
    # sooner than the values themselves do.
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    # clamp_min, not `+eps`: an all-zero row has amax 0 and would otherwise
    # divide by zero -> NaN through the whole GEMM. The floor value is
    # irrelevant (the row is zero either way) but it must be strictly positive
    # so the returned scale stays a valid dequantisation factor.
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    q = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale


# One inductor kernel per shape; ~5.4x over eager (0.527 -> 0.097 ms at
# [10368, 3072]) because eager materialises an fp32 copy of x. dynamic=True
# would compile once but runs 1.85x slower, and the shape set here is a handful
# -- see prewarm_quantizer.
#
# The catch: dynamo's recompile_limit defaults to 8, and past it dynamo drops the
# frame to EAGER -- measured, and it is only a warning, so the fp8 path would
# quietly lose ~1.4s over a clip. The SR loop already reaches 5-7 shapes and a
# ragged resolution adds more. So we budget the static compiles ourselves and
# send the overflow to a SEPARATE dynamic=True graph: 1.85x slower than static
# but 2.9x faster than eager, and one compile for every remaining shape.
# Keeping them as two torch.compile objects (rather than letting dynamo switch
# the single frame to automatic-dynamic) is what preserves the static kernel for
# the hot shape -- a dynamic guard on the same frame would shadow it.
_QUANT_MAX_COMPILES = int(os.environ.get("FP8_QUANT_MAX_COMPILES", "32"))
_quantize_static = None
_quantize_dynamic = None
_static_shapes = set()


def quant_max_compiles():
    """How many distinct shapes get their own static kernel."""
    return _QUANT_MAX_COMPILES


def quantizer_stats():
    return {"static_shapes": len(_static_shapes),
            "budget": _QUANT_MAX_COMPILES,
            "dynamic_used": _quantize_dynamic is not None,
            "shapes": sorted(_static_shapes)}


def reset_quantizer_state():
    """Drop the compiled quantisers and the shape ledger (tests only)."""
    global _quantize_static, _quantize_dynamic
    _quantize_static = None
    _quantize_dynamic = None
    _static_shapes.clear()


def _raise_recompile_limit(n):
    """Let dynamo recompile up to our own budget instead of bailing at 8."""
    import torch._dynamo as dyn
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(dyn.config, attr) and getattr(dyn.config, attr) < n:
            setattr(dyn.config, attr, n)
    for attr in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(dyn.config, attr) and getattr(dyn.config, attr) < 8 * n:
            setattr(dyn.config, attr, 8 * n)


def quantize_rowwise(x):
    """[..., K] -> (fp8 [..., K], fp32 scale [..., 1]) such that q*scale ~= x."""
    global _quantize_static, _quantize_dynamic
    if os.environ.get("FP8_COMPILE_QUANT", "1") == "0":
        return _quantize_rowwise_impl(x)
    key = tuple(x.shape)
    if key in _static_shapes or len(_static_shapes) < _QUANT_MAX_COMPILES:
        if _quantize_static is None:
            _raise_recompile_limit(_QUANT_MAX_COMPILES)
            _quantize_static = torch.compile(_quantize_rowwise_impl, dynamic=False)
        _static_shapes.add(key)
        return _quantize_static(x)
    if _quantize_dynamic is None:
        _quantize_dynamic = torch.compile(_quantize_rowwise_impl, dynamic=True)
    return _quantize_dynamic(x)


def quantizer_shapes(token_counts, dim, ffn_dim, text_token_counts=()):
    """The exact (M, K) the converted Linears will see -- the prewarm's list.

    Enumerated rather than cross-producted from the modules' in_features: the
    text projections only ever see text-length inputs, so a cross product would
    add shapes like (512, ffn_dim) that never run, each costing a ~1.6 s compile
    and a slot in the static budget.
    """
    shapes = []
    for m in sorted({int(t) for t in token_counts}):
        shapes.append((m, int(dim)))        # q/k/v/o, cross q/o, ffn.0
        shapes.append((m, int(ffn_dim)))    # ffn.2, on the FFN hidden state
    for m in sorted({int(t) for t in text_token_counts}):
        shapes.append((m, int(dim)))        # cross_attn.k/v, on the text context
    return list(dict.fromkeys(shapes))


def prewarm_quantizer(shapes, device=None, dtype=torch.bfloat16):
    """Compile the quantiser for `shapes` now instead of mid-inference.

    Same argument as the block-grid kernel's prewarm: inductor's per-shape
    compile is ~1.6 s and lands inside the first DiT forwards otherwise, where
    it shows up as the fp8 path being slower than bf16 on a cold process.
    Returns the shapes it managed to compile.
    """
    if not fp8_supported(device):
        return []
    dev = torch.device(device if device is not None else "cuda")
    done = []
    for shape in shapes:
        try:
            quantize_rowwise(torch.zeros(*shape, device=dev, dtype=dtype))
            done.append(tuple(shape))
        except Exception:
            continue
    if done:
        torch.cuda.synchronize(dev)
    return done


class Fp8Linear(nn.Module):
    """Drop-in for a frozen inference-time nn.Linear, computed in fp8.

    The weight is quantised ONCE here; only the activation is quantised per
    call. Requires the LoRA to already be merged into the base weight (it is:
    inference_sr.py calls merge_and_unload before this runs), otherwise the
    adapter would be applied to a weight this module no longer owns.
    """

    def __init__(self, weight, bias=None):
        super().__init__()
        if weight.dim() != 2:
            raise ValueError(f"expected a 2D weight, got {tuple(weight.shape)}")
        self.out_features, self.in_features = weight.shape
        w_fp8, w_scale = _quantize_rowwise_impl(weight.detach())
        # _scaled_mm wants the rhs column-major. Keeping w_fp8 as [N, K] and
        # transposing at call time is a free stride flip, not a copy.
        self.register_buffer("weight_fp8", w_fp8, persistent=False)
        # [1, N] to broadcast down the output columns of an [M, N] result.
        self.register_buffer("weight_scale", w_scale.reshape(1, -1).contiguous(),
                             persistent=False)
        if bias is None:
            self.bias = None
        else:
            # _scaled_mm folds the bias into its epilogue, which both saves a
            # full-size read+write pass and rounds once instead of twice (it is
            # applied to the fp32 accumulator, before the bf16 store). It only
            # accepts Half/BFloat16, so an fp32 bias is narrowed here.
            b = bias.detach().clone()
            if b.dtype not in (torch.bfloat16, torch.float16):
                b = b.to(torch.bfloat16)
            self.register_buffer("bias", b, persistent=False)
        # models.py:608 / :930 read `self.o.weight.dtype` to cast their input to
        # the Linear's compute dtype. Keeping a zero-element stand-in lets those
        # call sites keep working with models.py untouched (so the training path
        # cannot regress), while shape 0 makes any attempt to actually use it as
        # a weight fail immediately rather than silently compute garbage.
        self.register_buffer("weight",
                             torch.empty(0, dtype=weight.dtype,
                                         device=weight.device),
                             persistent=False)

    @classmethod
    def from_linear(cls, lin):
        return cls(lin.weight, lin.bias).to(lin.weight.device)

    def _apply(self, fn, recurse=True):
        """Keep the quantised buffers in their own dtypes across `.to()`/`.cuda()`.

        `Module.to(dtype)` converts every FLOATING-POINT buffer, and fp8 is one:
        a stray `.to(torch.bfloat16)` downstream of the conversion would rewrite
        the e4m3 code words as though they were the real weight, and the model
        would keep running, on garbage. An fp32 cast would additionally leave a
        bias dtype _scaled_mm refuses. Restoring is exact -- every e4m3 value is
        representable in bf16/fp32 -- so this loses nothing.
        """
        # Restore the ORIGINAL tensors (on the new device) rather than casting the
        # converted ones back: a bf16 round-trip of the fp32 per-channel scale is
        # NOT lossless and shifts the output by ~1 ULP of the result.
        saved = {n: b for n, b in self._buffers.items()
                 if n in ("weight_fp8", "weight_scale", "bias") and b is not None}
        out = super()._apply(fn, recurse)
        for name, old in saved.items():
            new = out._buffers[name]
            if new is not None and new.dtype != old.dtype:
                out._buffers[name] = old.to(device=new.device)
        return out

    def forward(self, x):
        out_dtype = x.dtype
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        # _scaled_mm requires a contiguous row-major lhs; the reshape above is
        # a view for contiguous inputs and a copy otherwise.
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        xq, xs = quantize_rowwise(x2)
        out = torch._scaled_mm(
            xq, self.weight_fp8.t(),
            scale_a=xs, scale_b=self.weight_scale,
            bias=self.bias, out_dtype=torch.bfloat16)
        if out_dtype is not torch.bfloat16:
            out = out.to(out_dtype)
        return out.reshape(*lead, self.out_features)

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, fp8=e4m3-rowwise")


def _block_index(name):
    """'blocks.7.ffn.0' -> 7; None if the name is not inside a block."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
        return int(parts[1])
    return None


def convert_linears_to_fp8(model, targets=TARGETS_ALL, skip_first=0, skip_last=0,
                           blocks_attr="blocks"):
    """Swap the target Linears inside `model.blocks` for Fp8Linear, in place.

    `skip_first` / `skip_last` leave that many leading/trailing blocks in bf16 --
    the usual mitigation when fp8 costs too much quality, since the first
    blocks set up the feature scale and the last ones feed the output head.

    Returns the qualified names actually converted (empty if fp8 is
    unsupported, or on a second call -- the swap is idempotent).
    """
    if not fp8_supported(next(model.parameters(), torch.empty(0)).device
                         if any(True for _ in model.parameters()) else None):
        return []
    blocks = getattr(model, blocks_attr, None)
    if blocks is None:
        return []
    n = len(blocks)
    lo, hi = int(skip_first), n - int(skip_last)
    converted = []
    for i, block in enumerate(blocks):
        if not (lo <= i < hi):
            continue
        for suffix in targets:
            *path, leaf = suffix.split(".")
            parent = block
            for p in path:
                parent = getattr(parent, p, None) if not p.isdigit() else parent[int(p)]
                if parent is None:
                    break
            if parent is None:
                continue
            cur = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf, None)
            if not isinstance(cur, nn.Linear):
                continue                      # missing, or already an Fp8Linear
            new = Fp8Linear.from_linear(cur)
            if leaf.isdigit():
                parent[int(leaf)] = new
            else:
                setattr(parent, leaf, new)
            converted.append(f"{blocks_attr}.{i}.{suffix}")
    return converted
