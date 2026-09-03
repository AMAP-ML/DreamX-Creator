# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    from flash_attn import flash_attn_func as _flash_attn_func
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    _flash_attn_func = None
    FLASH_ATTN_2_AVAILABLE = False

try: # MagiAttention FFA (third block-grid window backend, attn_impl="magi")
    from magi_attention.functional import flex_flash_attn_func as _magi_ffa
    MAGI_ATTN_AVAILABLE = True
except Exception:  # ModuleNotFoundError or a partially-built extension
    _magi_ffa = None
    MAGI_ATTN_AVAILABLE = False

try:  # fused Triton kernels (fourth block-grid window backend, attn_impl="triton")
    from . import blockgrid_triton as _bgt
except ImportError:
    # This file can also be loaded as a LEAF module via spec_from_file_location
    # (no parent package to resolve a relative import against), so fall back to
    # loading the sibling by path. The leaf copy carries its own plan cache and
    # JIT cache.
    import importlib.util as _ilu
    import pathlib as _pl
    _bgt_spec = _ilu.spec_from_file_location(
        "_sr_dit_blockgrid_triton_leaf",
        _pl.Path(__file__).resolve().parent / "blockgrid_triton.py")
    _bgt = _ilu.module_from_spec(_bgt_spec)
    _bgt_spec.loader.exec_module(_bgt)

TRITON_AVAILABLE = _bgt.TRITON_AVAILABLE
# Same dict object as blockgrid_triton.TRITON_PLAN_CACHE, so external helpers can
# clear both through either reference.
_TRITON_PLAN_CACHE = _bgt.TRITON_PLAN_CACHE
# Looked up as a module global at call time so it can be swapped in tests.
_run_blockgrid_triton = _bgt.run_blockgrid_triton

import warnings
import math
from typing import NamedTuple
import torch.nn.functional as F
from einops import rearrange


__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """Unified wrapper that dispatches to FlashAttention v3, v2, or PyTorch SDPA."""
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    if q.device.type != 'cuda' or q.size(-1) > 256:
        # Fallback to PyTorch SDPA when FlashAttention constraints are not met.
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
        )
        return out.transpose(1, 2).contiguous()

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        # pack padded sequences into a single contiguous buffer
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale  # useful for qk_norm variants

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)
        # FA3 returns (output, softmax_lse)
        if isinstance(x, tuple):
            x = x[0]
        x = x.unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """User-facing attention helper that prefers FlashAttention if available."""
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None  # PyTorch SDPA currently lacks varlen kernel, so mask is dropped

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


# =========================================================================== #
# Causal block-grid (MAGI-style) mask-free window self-attention.              #
#                                                                              #
# Query blocks HARD-PARTITION the (f, h, w) latent grid into (block_t, bh, bw) #
# cuboids; each block attends its boundary-clamped (rh, rw) spatial            #
# neighborhood over a causal temporal span. Visibility comes from WHAT GETS    #
# ADDRESSED, so the path stays mask-free (no attn_mask, no wasted compute).    #
# Three interchangeable backends share one geometry spec:                      #
#   `gather` — per-window KV copies + SDPA/flash; autograd-safe, but the       #
#              backward graph holds O(Nw*Lkv) gathered KV,                     #
#   `flex`   — flex_attention + BlockMask over the flat [anchor | KV] axis;    #
#              a single K/V copy, the low-memory training backend,             #
#   `magi`   — MagiAttention flex_flash_attn_func over a rect range table on a #
#              block-packed sequence; the fastest backend.                     #
# =========================================================================== #


def _win_lin_indices(T, h, w, h_starts, w_starts, wh, ww, t_start: int = 0) -> torch.Tensor:
    """Flat token indices for every window over a (T, h, w) grid.

    Returns [Nw, T*wh*ww] with Nw = len(h_starts)*len(w_starts). Token order
    inside a window matches the natural (t, hh, ww) flatten so a single window
    covering the whole grid yields an identity gather.

    ``t_start`` shifts the temporal range to frames [t_start, t_start+T), which is
    what lets the chunk-causal path address an arbitrary frame span of the clip.
    """
    dh = torch.arange(wh)
    dw = torch.arange(ww)
    dt = torch.arange(t_start, t_start + T)
    h_idx = h_starts[:, None] + dh[None, :]          # [nh, wh]
    w_idx = w_starts[:, None] + dw[None, :]          # [nw, ww]
    spatial = (h_idx[:, None, :, None] * w + w_idx[None, :, None, :]).reshape(-1, wh * ww)
    full = dt[None, :, None] * (h * w) + spatial[:, None, :]   # [Nw, T, wh*ww]
    return full.reshape(spatial.shape[0], T * wh * ww)


_WIN_BLOCKGRID_IDX_CACHE: dict = {}


def _blockgrid_neighbours(nb, b, radius):
    """Clamped neighbour block range [lo, hi] (INCLUSIVE) of block ``b``.

    THE definition of "block b's neighbourhood", shared by every backend:
    [b-radius, b+radius] clamped to [0, nb-1]. Clamping (rather than dropping)
    keeps the set CONTIGUOUS, which is what makes each neighbourhood a single KV
    interval for gather/flex and a single rectangle per group for magi.
    """
    return max(0, b - radius), min(nb - 1, b + radius)


def _blockgrid_axis_blocks(size, block, radius):
    """Per-block (q_start, q_len, kv_start, kv_len) along one axis (MAGI-style).

    THE spatial rule. Blocks tile [0, size) contiguously: block ``b`` occupies
    [b*block, min((b+1)*block, size)) — the tail block is smaller when ``block``
    does not divide ``size`` (MAGI ``_dim_block_sizes``). The neighbour block set
    (``_blockgrid_neighbours``) is a contiguous run, so its KV token span is a
    single interval [lo*block, min((hi+1)*block, size)).

    Returns a list of length ``nb = ceil(size/block)``. Integer-only.
    """
    nb = (size + block - 1) // block
    out = []
    for b in range(nb):
        q0, q1 = b * block, min((b + 1) * block, size)
        lo, hi = _blockgrid_neighbours(nb, b, radius)
        k0, k1 = lo * block, min((hi + 1) * block, size)
        out.append((q0, q1 - q0, k0, k1 - k0))
    return out


def _anchor_grid_stride(h, w, bh, bw, anchor_hw):
    """(s_h, s_w) of a COARSER anchor grid, with the divisibility contract enforced.

    The native LQ-anchor keeps the resolution it was harvested at: the anchor grid
    is ``(h//s_h, w//s_w)`` while the queries stay on ``(h, w)``. A query block's
    anchor extent is then its own HR neighbourhood extent divided by the stride —
    the same region of the PICTURE, ``s_h*s_w`` fewer tokens.

    That division has to be exact, and not only for tidiness: the mask-free
    contract requires every window inside one bucket to have the SAME anchor
    length. Bucketing is by ``khL*kwL``, and the anchor length is
    ``khL*kwL/(s_h*s_w)``, which is an integer and constant per bucket exactly
    when ``s_h`` divides both ``h`` and ``bh`` (and the w twins) — that makes
    every block extent a multiple of the stride. Violations raise instead of
    rounding, since a ragged ``Lsp`` would silently produce a plausible wrong
    number.
    """
    if anchor_hw is None:
        return 1, 1
    h_a, w_a = int(anchor_hw[0]), int(anchor_hw[1])
    if h_a <= 0 or w_a <= 0:
        raise ValueError(f"anchor_hw must be positive, got {(h_a, w_a)}.")
    if h % h_a or w % w_a:
        raise ValueError(
            f"anchor grid {(h_a, w_a)} must divide the query token grid "
            f"{(h, w)}: got h % h_a = {h % h_a}, w % w_a = {w % w_a}. The "
            f"anchor is a SHRUNKEN copy of the same extent, so its grid has to "
            f"be an integer factor of the query grid.")
    s_h, s_w = h // h_a, w // w_a
    if bh % s_h or bw % s_w:
        raise ValueError(
            f"anchor stride {(s_h, s_w)} must divide the query block size "
            f"{(bh, bw)}: got bh % s_h = {bh % s_h}, bw % s_w = {bw % s_w}. "
            f"Otherwise one bucket mixes different anchor lengths (ragged Lsp), "
            f"which the mask-free window path cannot express.")
    return s_h, s_w


def _blockgrid_extent_tables(size, block, radius, device):
    """Per-POSITION spatial KV extent [k0, k1) of the position's OWNING block.

    The same spatial rule as ``_blockgrid_axis_blocks``, transposed from
    per-block to per-position: rows/cols tile contiguously, so
    position -> block -> extent is a lookup table. The flex backend needs it per
    query token (its mask_mod is evaluated per (q, kv) pair) rather than per
    block; deriving it here keeps the spatial rule single-sourced.
    """
    k0 = torch.zeros(size, dtype=torch.long, device=device)
    k1 = torch.zeros(size, dtype=torch.long, device=device)
    for q0, qL, kk0, kkL in _blockgrid_axis_blocks(size, block, radius):
        k0[q0:q0 + qL] = kk0
        k1[q0:q0 + qL] = kk0 + kkL
    return k0, k1


class _BlockGridSpec(NamedTuple):
    """One block-grid window geometry — the single source of truth for VISIBILITY.

    The three backends (gather / flex / magi) differ only in HOW they execute a
    geometry; WHAT each query sees comes from here, so a rule is stated once:

      - spatial: :meth:`h_blocks` / :meth:`w_blocks` (``_blockgrid_axis_blocks``),
      - temporal: :meth:`temporal_runs`, which carries BOTH supported rules.

    Fields:
        f_cur, f_kv: query / KV latent frame counts (f_kv > f_cur ⇔ KV cache).
        h, w: token grid.
        block_t, bh, bw: query block size along (t, row, col).
        rh, rw: spatial neighbourhood radius in block units.
        hist: chunk-anchored history depth in FRAMES, ``(2*radius_t+1)*block_t``.
            ``None`` selects the legacy derived-``rt`` rule instead.
        rt: legacy temporal radius in ``block_t`` units, derived by the caller
            from the span it already receives (``causal_kv_span//block_t - 1``
            parallel, ``f_kv//block_t - 1`` streaming). Ignored when ``hist``
            is set.
        chunk_frames: frames per causal chunk (``causal_chunk_frames``), the
            unit the chunk-anchored rule partitions the query time axis by.
        streaming: a KV cache and/or an LQ anchor is present, i.e. this forward
            is one step of a rollout rather than a full-sequence pass. Causality
            is then structural (only past frames are held), so the whole current
            chunk is ONE query run.
        query_grain: "grid" (the temporal axis is partitioned into query blocks)
            or "chunk" (one run = the whole current chunk, the LQ-anchor gather).
            There is deliberately no per-frame grain: a frame-aligned anchor
            changes which anchor COLUMNS a query frame sees, never its HR
            temporal span, so measuring ``hist`` from a single frame instead of
            from the chunk would shorten the span the other backends use.
        anchor_hw: the LQ-anchor's OWN token grid ``(h_a, w_a)`` when it is
            coarser than the query grid (the native LQ-anchor, ``--lq_anchor_
            native_scheme true``), else ``None`` — which is every other caller,
            including scheme "repeat", and takes the pre-existing code path
            bit-for-bit. See :func:`_anchor_grid_stride`.
    """

    f_cur: int
    f_kv: int
    h: int
    w: int
    block_t: int
    bh: int
    bw: int
    rh: int
    rw: int
    hist: int | None = None
    rt: int = 0
    chunk_frames: int | None = None
    streaming: bool = False
    query_grain: str = "grid"
    anchor_hw: tuple | None = None

    def h_blocks(self):
        return _blockgrid_axis_blocks(self.h, self.bh, self.rh)

    def w_blocks(self):
        return _blockgrid_axis_blocks(self.w, self.bw, self.rw)

    def anchor_stride(self):
        """(s_h, s_w) — (1, 1) unless ``anchor_hw`` is a coarser grid."""
        return _anchor_grid_stride(self.h, self.w, self.bh, self.bw,
                                   self.anchor_hw)

    def anchor_h_blocks(self):
        """Per-block ANCHOR row extent: the HR extent divided by the stride.

        Deliberately derived from :meth:`h_blocks` rather than re-tiling the
        anchor grid on its own. The two agree exactly (``_blockgrid_axis_blocks(
        h//s, bh//s, rh)`` has the same block count and the same clamped
        neighbourhood), but dividing keeps the HR↔anchor pairing structural: the
        i-th anchor extent belongs to the i-th HR block by construction, so the
        two can never be zipped out of order.
        """
        s_h = self.anchor_stride()[0]
        return [(q0, qL, k0 // s_h, kL // s_h) for (q0, qL, k0, kL) in self.h_blocks()]

    def anchor_w_blocks(self):
        s_w = self.anchor_stride()[1]
        return [(q0, qL, k0 // s_w, kL // s_w) for (q0, qL, k0, kL) in self.w_blocks()]

    @property
    def anchor_grid(self):
        """The anchor's (h_a, w_a) — the query grid itself when ``anchor_hw`` is None."""
        if self.anchor_hw is None:
            return int(self.h), int(self.w)
        return int(self.anchor_hw[0]), int(self.anchor_hw[1])

    @property
    def anchor_frame_tokens(self) -> int:
        """``L_a`` — anchor tokens per frame, i.e. the HR offset in the extended KV."""
        h_a, w_a = self.anchor_grid
        return h_a * w_a

    @property
    def one_query_run(self) -> bool:
        """True when the query time axis is NOT partitioned into blocks/chunks.

        Streaming (cache and/or anchor), an explicitly coarse grain, and a KV
        grid longer than the query grid all mean the same thing: the current
        chunk is a single query run that attends a tail window of history.
        ``f_kv != f_cur`` implies a cache, hence ``streaming`` — it is listed for
        callers that build a plan without saying which regime they are in.
        """
        return (bool(self.streaming) or self.query_grain != "grid"
                or self.f_kv != self.f_cur)

    def temporal_runs(self):
        """(q_f0, q_f1, kv_f0, kv_f1) frame runs — THE temporal rule.

        Both supported rules live here, and only here; all three backends read
        this method, so they cannot drift apart:

          - ``hist`` set — CHUNK-ANCHORED: a query in chunk [c0, c1) sees
            ``[max(0, c0-hist), c1)``, i.e. its whole own chunk plus ``hist``
            history frames. Streaming and parallel receptive fields are
            frame-identical by construction, which is what the magi backend's
            b=(t,bh,bw) / r=(rt,rh,rw) parametrization means.
          - ``hist`` None — LEGACY derived-``rt``: query block ``i`` (``block_t``
            frames) sees ``[max(0, i-rt)*block_t, (i+1)*block_t)``; a single
            query run sees the tail window ``[max(0, f_kv-(rt+1)*block_t), f_kv)``.
            Both give the same frame-for-frame receptive field as the span the
            caller derived ``rt`` from.

        The runs PARTITION the query frame axis in ascending order (checked
        below): the gather plan turns them into (start, extent) index rows, magi
        into q-runs of rectangles, and flex scatters them onto query rows.
        """
        f_cur, f_kv = int(self.f_cur), int(self.f_kv)
        if self.query_grain not in ("grid", "chunk"):
            raise ValueError(
                f"query_grain must be 'grid'|'chunk', got "
                f"{self.query_grain!r}.")
        if self.hist is not None:
            hist = int(self.hist)
            cf = int(self.chunk_frames) if self.chunk_frames else max(1, f_cur)
            if self.one_query_run or f_cur <= cf:
                runs = [(0, f_cur, max(0, f_kv - f_cur - hist), f_kv)]
            else:
                runs = []
                for c0 in range(0, f_cur, cf):
                    c1 = min(c0 + cf, f_cur)
                    runs.append((c0, c1, max(0, c0 - hist), min(f_kv, c1)))
        else:
            bt, rt = int(self.block_t), int(self.rt)
            if self.one_query_run:
                runs = [(0, f_cur, max(0, f_kv - (rt + 1) * bt), f_kv)]
            elif f_cur < bt:
                # Sub-block query grid (e.g. a 1-frame anchor harvest): one
                # query block attending everything available instead of an
                # empty span.
                runs = [(0, f_cur, 0, f_kv)]
            else:
                runs = [(i * bt, (i + 1) * bt,
                         max(0, i - rt) * bt, (i + 1) * bt)
                        for i in range(f_cur // bt)]
        covered = sum(q1 - q0 for (q0, q1, _, _) in runs)
        if covered != f_cur or runs[0][0] != 0 or runs[-1][1] != f_cur:
            raise ValueError(
                f"temporal runs {runs} do not partition the {f_cur} query "
                f"frames (covered {covered}); block_t={self.block_t} must "
                f"divide f_cur in the partitioned regimes.")
        return runs


def _get_win_meta_blockgrid(f_cur, f_kv, h, w, block_t, block_hw, block_radius_hw,
                            rt, device, chunk_query=False,
                            hist=None, chunk_frames=None, streaming=False,
                            anchor_hw=None):
    """Gather plan for the causal block-grid (MAGI-style) window path.

    Query blocks HARD-PARTITION the (f_cur, h, w) grid into cuboids of size
    (block_t, bh, bw). Each query block attends the CAUSAL neighborhood cuboid
    dt in [-rt, 0], dh in [-rh, rh], dw in [-rw, rw] over the (f_kv, h, w) KV
    grid, dropping out-of-bounds neighbors (validity, not clamp of the query
    block). Because block tiling is contiguous, each neighborhood is a single
    KV cuboid, so ``_win_lin_indices`` builds both the Q and KV index rows from
    one (start, extent) each — no new low-level index routine is needed.

    The temporal span comes from ``_BlockGridSpec.temporal_runs`` (both rules,
    shared with the flex and magi backends); ``chunk_query`` selects that spec's
    coarse query grain, which the LQ-anchor gather needs: the query temporal
    block is the WHOLE current chunk (frames [0, f_cur)) — also when
    f_kv == f_cur (no-cache first chunk), so a chunk-aligned anchor stays fully
    bidirectional inside the chunk instead of falling into the parallel
    chunk-causal tiling. Query index rows are then frame-major inside a window
    ([f_cur, qhL*qwL]), which is what lets a frame-aligned anchor serve one
    frame at a time out of this ONE plan.

    Query blocks partition tokens, so the owner map is the identity in
    window-major order and the caller scatters by ``q_lin_flat`` directly.
    Blocks with different (Lq, Lkv) (edge / tail blocks, and early causal
    chunks) are bucketed into separate groups so every batched attention stays
    uniform and mask-free.

    Returns a list of groups:
    {q_lin_flat, kv_lin_flat, kv_sp_flat, Lsp, Nw, Lq, Lkv}. ``kv_sp_flat`` is
    the spatial-only ([0, h*w)) copy of the KV neighborhood used by the windowed
    LQ-anchor, and is None outside the chunk_query regime.

    ``anchor_hw=(h_a, w_a)`` builds ``kv_sp_flat`` on the ANCHOR's own coarser
    grid instead (values in [0, h_a*w_a), the HR extent divided by the stride) —
    the native LQ-anchor. ``None`` is every other caller and leaves the rows
    bit-identical to before.
    """
    bh, bw = int(block_hw[0]), int(block_hw[1])
    rh, rw = int(block_radius_hw[0]), int(block_radius_hw[1])
    grain = "chunk" if chunk_query else "grid"
    key = (f_cur, f_kv, h, w, block_t, bh, bw, rh, rw,
           # `rt` is the legacy rule's radius; under the chunk-anchored rule
           # (`hist`) it is unused, so it must not split the cache.
           None if hist is not None else rt,
           grain, hist,
           None if chunk_frames is None else int(chunk_frames), bool(streaming),
           device.type, getattr(device, "index", None),
           # Only the anchor rows depend on it, and only in the chunk_query
           # regime — but they live in the same cached groups.
           None if anchor_hw is None else (int(anchor_hw[0]), int(anchor_hw[1])))
    groups = _WIN_BLOCKGRID_IDX_CACHE.get(key)
    if groups is not None:
        return groups

    spec = _BlockGridSpec(
        f_cur=f_cur, f_kv=f_kv, h=h, w=w, block_t=int(block_t),
        bh=bh, bw=bw, rh=rh, rw=rw,
        hist=None if hist is None else int(hist), rt=int(rt),
        chunk_frames=None if chunk_frames is None else int(chunk_frames),
        streaming=bool(streaming), query_grain=grain,
        anchor_hw=None if anchor_hw is None else (int(anchor_hw[0]),
                                                 int(anchor_hw[1])))
    h_blocks = spec.h_blocks()
    w_blocks = spec.w_blocks()

    # THE temporal rule, as (q_t0, q_tL, kv_t0, kv_tL) for `_win_lin_indices`.
    t_pairs = [(q0, q1 - q0, k0, k1 - k0)
               for (q0, q1, k0, k1) in spec.temporal_runs()]

    # A SPATIAL-ONLY copy of the KV neighborhood (values in [0, h*w)) is what the
    # windowed LQ-anchor gathers: the anchor lives on its own frame axis, so the
    # caller re-offsets these rows per anchor frame. Only built for the anchor
    # regime (chunk_query): it has a single t_pair, so ``khL*kwL`` is constant
    # inside a bucket. The generic regimes can mix different (kv_tL, khL*kwL)
    # factorisations in one bucket, where a spatial cat would be ragged — and
    # they never use an anchor.
    want_sp = bool(chunk_query)
    # The anchor's spatial rows live on the anchor grid, which is the query grid
    # unless `anchor_hw` shrinks it. Paired index-by-index with the HR blocks
    # (`anchor_h_blocks` divides the very same extents), so a bucket's anchor
    # length stays `khL*kwL/(s_h*s_w)` — constant, hence mask-free.
    a_h, a_w = spec.anchor_grid
    hb_a = spec.anchor_h_blocks() if anchor_hw is not None else h_blocks
    wb_a = spec.anchor_w_blocks() if anchor_hw is not None else w_blocks
    buckets: dict = {}
    for (q_t0, q_tL, kv_t0, kv_tL) in t_pairs:
        for hi, (qh0, qhL, kh0, khL) in enumerate(h_blocks):
            for wi, (qw0, qwL, kw0, kwL) in enumerate(w_blocks):
                q_row = _win_lin_indices(
                    q_tL, h, w, torch.tensor([qh0]), torch.tensor([qw0]),
                    qhL, qwL, t_start=q_t0)
                kv_row = _win_lin_indices(
                    kv_tL, h, w, torch.tensor([kh0]), torch.tensor([kw0]),
                    khL, kwL, t_start=kv_t0)
                bkey = (q_tL * qhL * qwL, kv_tL * khL * kwL)
                buckets.setdefault(bkey, ([], [], []))
                buckets[bkey][0].append(q_row)
                buckets[bkey][1].append(kv_row)
                if want_sp:
                    akh0, akhL = hb_a[hi][2], hb_a[hi][3]
                    akw0, akwL = wb_a[wi][2], wb_a[wi][3]
                    buckets[bkey][2].append(_win_lin_indices(
                        1, a_h, a_w, torch.tensor([akh0]), torch.tensor([akw0]),
                        akhL, akwL))

    groups = []
    for (Lq, Lkv), (q_rows, kv_rows, sp_rows) in buckets.items():
        q_lin = torch.cat(q_rows, dim=0)     # [Nw, Lq]
        kv_lin = torch.cat(kv_rows, dim=0)   # [Nw, Lkv]
        sp_lin = torch.cat(sp_rows, dim=0) if want_sp else None   # [Nw, Lsp]
        groups.append({
            "q_lin_flat": q_lin.reshape(-1).to(device),
            "kv_lin_flat": kv_lin.reshape(-1).to(device),
            "kv_sp_flat": None if sp_lin is None else sp_lin.reshape(-1).to(device),
            "Lsp": 0 if sp_lin is None else int(sp_lin.shape[1]),
            "Nw": int(q_lin.shape[0]), "Lq": int(Lq), "Lkv": int(Lkv),
        })
    _WIN_BLOCKGRID_IDX_CACHE[key] = groups
    return groups


def _win_use_flash(attn_impl, qg):
    """Whether to route windowed attention through flash_attn (CUDA + fp16/bf16)."""
    if attn_impl == "sdpa":
        return False
    if _flash_attn_func is None:
        return False
    if not qg.is_cuda or qg.dtype not in (torch.float16, torch.bfloat16):
        return False
    return attn_impl in ("auto", "flash")


def _win_attend(qg, kg, vg, attn_impl):
    """Mask-free attention over gathered windows.
    qg [Nw, Lq, n, d], kg/vg [Nw, Lkv, n, d] -> [Nw, Lq, n, d]."""
    if _win_use_flash(attn_impl, qg):
        # flash_attn expects [batch, seqlen, heads, dim] == the gather layout
        return _flash_attn_func(qg, kg, vg, causal=False)
    return F.scaled_dot_product_attention(
        qg.transpose(1, 2), kg.transpose(1, 2), vg.transpose(1, 2)).transpose(1, 2)


def _win_gather_attend(q_src, k_src, v_src, *, q_idx, kv_idx, Nw, Lq, Lkv,
                       n, d, attn_impl, win_chunk):
    """Gather windows -> batched attention -> window-major result [Nw*Lq, n, d].

    Query blocks HARD-PARTITION the token grid, so there is no owner priority to
    resolve: result row ``wi*Lq + j`` belongs to token ``q_idx[wi*Lq + j]`` and
    the caller scatters the whole thing with one ``index_copy_``. Numerically
    identical whether run one-shot or in ``win_chunk`` batches; the batching only
    bounds peak memory.
    """
    chunk = Nw if (win_chunk is None or win_chunk <= 0) else min(int(win_chunk), Nw)
    if chunk >= Nw:
        # one-shot: gather all windows, single batched attention
        qg = q_src.index_select(0, q_idx).view(Nw, Lq, n, d)
        kg = k_src.index_select(0, kv_idx).view(Nw, Lkv, n, d)
        vg = v_src.index_select(0, kv_idx).view(Nw, Lkv, n, d)
        return _win_attend(qg, kg, vg, attn_impl).reshape(Nw * Lq, n, d)
    # memory-bounded: `chunk` windows at a time into the same window-major buffer
    out = q_src.new_empty(Nw * Lq, n, d)
    for a in range(0, Nw, chunk):
        bnd = min(a + chunk, Nw)
        qb = q_src.index_select(0, q_idx[a * Lq:bnd * Lq]).view(bnd - a, Lq, n, d)
        kb = k_src.index_select(0, kv_idx[a * Lkv:bnd * Lkv]).view(bnd - a, Lkv, n, d)
        vb = v_src.index_select(0, kv_idx[a * Lkv:bnd * Lkv]).view(bnd - a, Lkv, n, d)
        out[a * Lq:bnd * Lq] = _win_attend(qb, kb, vb, attn_impl).reshape(
            (bnd - a) * Lq, n, d)
        del qb, kb, vb
    return out


def _anchor_kv_index(group, *, f_cur, L, win_scope, frame):
    """LQ-anchor KV columns for one window group -> ([Nw, a_len], a_len).

    THE anchor rule, shared by both aligns and both scopes (the flex/magi
    backends express the same set as a mask / rect table):

      - ``win_scope`` (scope="window"): the group's clamped-neighbourhood
        spatial rows — ``kv_sp_flat``, values in [0, L), the SAME extent as the
        group's HR KV — re-offset onto the anchor frame axis. At grid edges that
        neighbourhood is a shifted window, not the query block itself.
      - otherwise (scope="global"): every anchor token of the frame/chunk.
      - ``frame=fi`` (align="frame"): only anchor frame ``fi``, i.e. the query
        frame's OWN anchor (strict 1:1).
      - ``frame=None`` (align="chunk"): all ``f_cur`` anchor frames.

    Indices are relative to the anchor block of the extended KV, which the
    caller places FIRST ([anchor | HR]), so no offset is added here.

    ``L`` is the ANCHOR's tokens-per-frame, which is the HR ``h*w`` for every
    caller except the native LQ-anchor (where ``kv_sp_flat`` was already built on
    the coarser grid, so this function needs no other change).
    """
    Nw = group["Nw"]
    dev = group["q_lin_flat"].device
    if win_scope:
        sp = group["kv_sp_flat"].view(Nw, group["Lsp"])       # [Nw, Lsp] in [0, L)
        if frame is None:
            off = torch.arange(f_cur, device=dev) * L
            a_len = f_cur * group["Lsp"]
            return (sp[:, None, :] + off[None, :, None]).reshape(Nw, a_len), a_len
        return sp + frame * L, group["Lsp"]
    if frame is None:
        return torch.arange(f_cur * L, device=dev).expand(Nw, -1), f_cur * L
    return (torch.arange(frame * L, (frame + 1) * L, device=dev).expand(Nw, -1),
            L)


def _blockgrid_gather_attend(spec, q_i, full_k, full_v, *, anchor_k=None,
                             anchor_v=None, anchor_align="frame",
                             anchor_window_scope="window", attn_impl="auto",
                             win_chunk=None):
    """Gather execution of ONE sample's block-grid windows -> [s_cur, n, d].

    Materializes per-window KV copies and scatters the window-major result back
    with ``index_copy_`` (query blocks partition the grid, so the owner map is
    the identity). Autograd-safe: ``index_copy_`` is differentiable and every
    pass writes a FRESH ``_win_gather_attend`` result, so no buffer a saved
    graph node points at is ever overwritten (grads match the flex backend to
    ~2e-06). What it costs in
    training is memory, not correctness — the gathered KV stays live for the
    backward, which is why `flex`/`magi` exist.

    Visibility comes entirely from ``spec`` via ``_get_win_meta_blockgrid``;
    the anchor columns come from ``_anchor_kv_index``. Both aligns share ONE
    chunk-grain plan — the align only selects the anchor columns and how many
    query rows each pass serves: align="chunk" is a single pass with the whole
    anchor chunk visible, align="frame" walks the chunk's frames, slicing that
    frame's q rows out of the same plan and pairing them with that frame's own
    anchor.
    """
    f_cur, f_kv, h, w = spec.f_cur, spec.f_kv, spec.h, spec.w
    L = h * w
    # Anchor tokens per frame. Equal to L unless the anchor kept its own coarser
    # grid (native LQ-anchor), in which case the HR columns of the extended KV
    # shift by f_cur*L_a rather than f_cur*L.
    L_a = spec.anchor_frame_tokens
    n, d = q_i.shape[1], q_i.shape[2]
    anchor_given = anchor_k is not None
    frame_align = (anchor_align == "frame")
    win_scope = (anchor_window_scope == "window")
    out_i = q_i.new_empty(f_cur * L, n, d)

    groups = _get_win_meta_blockgrid(
        f_cur, f_kv, h, w, spec.block_t, (spec.bh, spec.bw), (spec.rh, spec.rw),
        spec.rt, q_i.device, chunk_query=anchor_given,
        hist=spec.hist, chunk_frames=spec.chunk_frames,
        streaming=spec.streaming, anchor_hw=spec.anchor_hw)

    if not anchor_given:
        for g in groups:
            tmp = _win_gather_attend(
                q_i, full_k, full_v, q_idx=g["q_lin_flat"],
                kv_idx=g["kv_lin_flat"], Nw=g["Nw"], Lq=g["Lq"], Lkv=g["Lkv"],
                n=n, d=d, attn_impl=attn_impl, win_chunk=win_chunk)
            # query blocks partition tokens -> scatter by global token id
            out_i.index_copy_(0, g["q_lin_flat"], tmp)
        return out_i

    # The anchor chunk is prepended to the KV and never enters the cache; HR KV
    # indices therefore shift by the anchor's f_cur*L columns.
    k_ext = torch.cat([anchor_k, full_k], dim=0)
    v_ext = torch.cat([anchor_v, full_v], dim=0)
    for g in groups:
        # Chunk-grain q rows are frame-major inside a window, so align="frame"
        # SLICES frame `fi` out of this plan (view [Nw, f_cur, Lq/f_cur]) instead
        # of building a second plan with a 1-frame query span — which would make
        # `temporal_runs` measure the history from one frame rather than from the
        # chunk, i.e. hand gather a shorter KV span than flex/magi.
        Lq = g["Lq"] // f_cur if frame_align else g["Lq"]
        q_rows = g["q_lin_flat"].view(g["Nw"], f_cur, Lq) if frame_align else \
            g["q_lin_flat"].view(g["Nw"], Lq)
        kv_span = g["kv_lin_flat"].view(g["Nw"], g["Lkv"]) + f_cur * L_a
        for frame in (range(f_cur) if frame_align else (None,)):
            a_idx, a_len = _anchor_kv_index(
                g, f_cur=f_cur, L=L_a, win_scope=win_scope, frame=frame)
            q_idx = (q_rows if frame is None else q_rows[:, frame]).reshape(-1)
            kv_idx = torch.cat([a_idx, kv_span], dim=1).reshape(-1)
            tmp = _win_gather_attend(
                q_i, k_ext, v_ext, q_idx=q_idx, kv_idx=kv_idx, Nw=g["Nw"],
                Lq=Lq, Lkv=a_len + g["Lkv"], n=n, d=d,
                attn_impl=attn_impl, win_chunk=win_chunk)
            out_i.index_copy_(0, q_idx, tmp)
    return out_i


#: Every backend selector the block-grid op accepts. An unrecognised string used
#: to fall through to the gather path, which silently reports gather's numbers
#: under another backend's label — the one failure mode a benchmark cannot see.
BLOCKGRID_IMPLS = ("auto", "flash", "sdpa", "flex", "magi", "triton")


def _blockgrid_triton_attend(spec, q_i, full_k, full_v, *, anchor_k=None,
                             anchor_v=None, anchor_align="frame",
                             anchor_window_scope="window"):
    """Fused execution of ONE sample's block-grid windows -> [s_cur, n, d].

    Gathers nothing: one Triton program per (window, m-tile, head) streams its
    own KV cuboid straight from the grid with an online softmax and stores its
    output tokens in place, with the LQ anchor fused into the same launch. The
    extended KV is the SAME ``cat([anchor, full])`` layout the gather backend
    builds, so ``HR_BASE = f_cur*L``; visibility comes entirely from ``spec``,
    via :func:`blockgrid_triton.build_plan` reading ``temporal_runs`` /
    ``h_blocks`` / ``w_blocks``.

    Unsupported cases RAISE rather than degrade to another backend — a silent
    fallback here would be reported as a triton measurement:

      * no triton / not CUDA — there is no CPU path;
      * autograd — the kernels write a raw buffer and record nothing, so a
        backward would find no path to q/k/v at all. Training uses
        ``flex``/``magi`` (the gather path is autograd-safe, this one is not);
      * ``anchor_window_scope="global"`` — the params row carries ONE spatial
        extent, shared by the anchor segment and the HR cuboid, so a
        frame-global anchor is not expressible. Production is "window"
        everywhere (the inference default and every block-grid config);
      * a head dim that is not a power of two (``tl.arange(0, D)``).
    """
    f_cur, h, w = spec.f_cur, spec.h, spec.w
    L = h * w
    n, d = q_i.shape[1], q_i.shape[2]
    anchor_given = anchor_k is not None

    if not TRITON_AVAILABLE:
        raise RuntimeError(
            'attn_impl="triton" needs the triton package; use "flex", "magi" '
            'or the gather backends ("auto"/"flash"/"sdpa").')
    if not q_i.is_cuda:
        raise RuntimeError(
            'attn_impl="triton" is CUDA-only; use "sdpa" on CPU.')
    if d & (d - 1):
        raise ValueError(
            f'attn_impl="triton" needs a power of two head dim, got {d}.')
    if torch.is_grad_enabled() and any(
            t is not None and t.requires_grad
            for t in (q_i, full_k, full_v, anchor_k, anchor_v)):
        raise RuntimeError(
            'attn_impl="triton" is forward-only (the fused kernels record '
            "nothing for autograd), but grad is enabled and an input requires "
            'grad. Train with attn_impl="flex" or "magi".')
    if anchor_given and anchor_window_scope != "window":
        raise ValueError(
            f'attn_impl="triton" only supports anchor_window_scope="window", '
            f"got {anchor_window_scope!r}: the fused params row carries one "
            f"spatial extent for both the anchor and the HR cuboid.")

    anchor_mode = (None if not anchor_given
                   else ("frame" if anchor_align == "frame" else "chunk"))
    plan = _bgt.cached_plan(spec, anchor_mode=anchor_mode)
    if anchor_given:
        # Never cached, and prepended so HR indices shift by f_cur*L — identical
        # to the gather backend's k_ext/v_ext.
        k_ext = torch.cat([anchor_k, full_k], dim=0)
        v_ext = torch.cat([anchor_v, full_v], dim=0)
    else:
        k_ext, v_ext = full_k, full_v
    # The kernels address tokens as flat*(n*d) + head*d + dim, so a stride-1
    # head dim over a contiguous token axis is part of the contract.
    out_i = q_i.new_empty(f_cur * L, n, d)
    return _run_blockgrid_triton(
        q_i.contiguous(), k_ext.contiguous(), v_ext.contiguous(), out_i,
        plan=plan, f_cur=f_cur, h=h, w=w, n=n, d=d, anchor_mode=anchor_mode)


def prewarm_blockgrid_triton(*, h, w, bt, bh, bw, rh, rw, n, d, variants,
                             block_radius_t=None, f_kv=None, device=None,
                             dtype=torch.bfloat16, anchor_hw=None):
    """JIT-compile the fused kernels for one geometry before inference starts.

    A cold process pays the compile inside its FIRST DiT forwards — 2263 ms for
    the frame kernel plus 1355 ms for the plain one at the shipped geometry,
    serial and GPU-idle — which is enough to turn the fused path's per-call win
    into an end-to-end loss (23.18 s vs 18.63 s of DiT; 16.13 s once warm).
    Launching the same specializations here moves that bill to setup; triton's
    on-disk cache then cuts it to ~0.5 s from the second process onward.

    ``variants`` is the list of ``(f_cur, anchor_mode)`` forwards the caller will
    actually make (``pipeline_sr.causal_inference.triton_prewarm_variants``
    derives it), and it has to be the real ones: the specialization does not
    depend on ``f_kv`` — the streaming ramp is free, every extent rides in the
    runtime params row — but it DOES depend on ``f_cur`` through triton's
    ``M_TILES == 1`` scalar specialization, which is exactly what makes the
    1-frame anchor harvest a second kernel. ``anchor_mode`` selects between the
    two kernels, and ``dtype`` is part of the launch signature, so the caller
    passes production ``h, w, n, d, dtype`` too.

    ``anchor_hw`` is the native LQ-anchor's own token grid (``None`` = the anchor
    shares the HR grid). It has to be the production value: ``HR_BASE`` becomes
    ``f_cur*h_a*w_a`` instead of ``f_cur*h*w``, and triton specializes on scalar
    ARGUMENTS — a different divisibility class there is a different compile, so
    warming the HR variant would leave the coarse one to the first forward.

    ``block_radius_t`` is the op's ``block_radius_t`` (``window_radius_t``), i.e.
    the chunk-anchored rule; ``None`` reproduces the legacy derived-``rt`` rule.
    It only moves ``kv_t0``/``kv_tL`` inside the params row, but the spec is built
    with the production rule anyway so the plan the prewarm launches is the plan
    shape the run launches.

    Returns the variants actually warmed — ``[]`` when the path cannot run here
    (no triton, no CUDA). A variant the planner or the launcher rejects is
    skipped rather than raised (prewarming must never abort a run) but is
    ``warnings.warn``-ed: a silently skipped variant puts the cold compile back
    inside the forwards, which is the one thing this function exists to prevent.
    """
    if not TRITON_AVAILABLE:
        return []
    dev = torch.device(device if device is not None else "cuda")
    if dev.type != "cuda":
        return []
    L, warmed = h * w, []
    a_hw = (None if anchor_hw is None or (int(anchor_hw[0]), int(anchor_hw[1])) == (h, w)
            else (int(anchor_hw[0]), int(anchor_hw[1])))
    L_a = L if a_hw is None else a_hw[0] * a_hw[1]
    for f_cur, anchor_mode in variants:
        f_cur = int(f_cur)
        # Smallest legal KV: f_kv does not affect the specialization, so the
        # dummy tensors stay small (one chunk, not a full ramped cache).
        fkv = int(f_kv) if f_kv is not None else max(f_cur, int(bt))
        a = f_cur if anchor_mode is not None else 0
        try:
            spec = _BlockGridSpec(
                f_cur=f_cur, f_kv=fkv, h=h, w=w, block_t=int(bt),
                bh=int(bh), bw=int(bw), rh=int(rh), rw=int(rw),
                hist=(None if block_radius_t is None
                      else (2 * int(block_radius_t) + 1) * int(bt)),
                rt=max(0, fkv // int(bt) - 1), chunk_frames=None,
                streaming=True, anchor_hw=a_hw)
            q = torch.zeros(f_cur * L, n, d, device=dev, dtype=dtype)
            kv = torch.zeros(a * L_a + fkv * L, n, d, device=dev, dtype=dtype)
            _run_blockgrid_triton(
                q, kv, kv, torch.empty_like(q),
                plan=_bgt.cached_plan(spec, anchor_mode=anchor_mode),
                f_cur=f_cur, h=h, w=w, n=n, d=d, anchor_mode=anchor_mode)
            warmed.append((f_cur, anchor_mode))
        except Exception as e:                          # noqa: BLE001
            warnings.warn(
                f"triton block-grid prewarm skipped variant "
                f"(f_cur={f_cur}, anchor_mode={anchor_mode!r}): {type(e).__name__}: "
                f"{e}. That kernel will be JIT-compiled inside the first forward "
                f"that needs it.")
    if warmed:
        torch.cuda.synchronize(dev)
    return warmed


def _truncate_kv_cache(full_k, full_v, kv_len, L, f_kv):
    """Tail-truncate the returned KV cache to the last ``kv_len`` latent frames.

    ``kv_len=None`` or a cache shorter than the budget returns the tensors
    unchanged (no copy). The LQ anchor is never part of ``full_k``/``full_v``,
    so it cannot leak into the cache.
    """
    if kv_len is None or f_kv <= kv_len:
        return full_k, full_v
    keep = int(kv_len) * L
    return full_k[-keep:], full_v[-keep:]


# ── flex_attention backend for block-grid windows ────────────────────────────
# Same pattern as the gather path — per owning query block: spatial KV extent
# from _blockgrid_axis_blocks, temporal span chunk-causal (parallel) or tail
# window (streaming/anchor), anchor frame/chunk align × window/global scope —
# but expressed as a BlockMask over ONE flat [anchor | KV] sequence, so nothing
# is gathered/duplicated: the training backward keeps a single K/V copy instead
# of O(Nw*Lkv) tokens. Numerically equivalent to the gather path.

_FLEX_ATTN = None


def _get_flex_attn():
    global _FLEX_ATTN
    if _FLEX_ATTN is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX_ATTN = torch.compile(flex_attention)
    return _FLEX_ATTN


def _blockgrid_flex_attend(spec, q_i, full_k, full_v, *,
                           anchor_k=None, anchor_v=None,
                           anchor_align="frame", anchor_window_scope="window"):
    """flex_attention execution of ONE sample's block-grid windows.

    Equivalent to the gather branch (incl. anchor semantics); returns
    ``[s_cur, n, d]``. Visibility comes entirely from ``spec``: the spatial
    extent from ``_blockgrid_extent_tables`` and the temporal span from
    ``temporal_runs`` (both rules), so this backend cannot drift from the
    others.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    dev = q_i.device
    f_cur, f_kv, h, w = spec.f_cur, spec.f_kv, spec.h, spec.w
    bh, bw, rh, rw = spec.bh, spec.bw, spec.rh, spec.rw
    L = h * w
    s_cur = f_cur * L
    n, d = q_i.shape[1], q_i.shape[2]
    anchor_given = anchor_k is not None
    # The anchor may live on its OWN, coarser grid (the native LQ-anchor). Then
    # the extended KV is [anchor(f_cur*L_a) | HR(f_kv*L)] with TWO moduli, and the
    # HR offset is L_a-based — the single most error-prone line in this backend,
    # since an HR-based offset reads plausible neighbour tokens instead of failing.
    a_h, a_w = spec.anchor_grid                  # == (h, w) when anchor_hw is None
    L_a = a_h * a_w
    s_h, s_w = spec.anchor_stride()              # also enforces the divisibility contract
    A = f_cur * L_a if anchor_given else 0

    # Per-position spatial KV extent of the OWNING query block.
    h_k0, h_k1 = _blockgrid_extent_tables(h, bh, rh, dev)
    w_k0, w_k1 = _blockgrid_extent_tables(w, bw, rw, dev)

    qpos = torch.arange(s_cur, device=dev)
    fi = qpos // L
    qr = (qpos % L) // w
    qc = qpos % w
    q_hk0, q_hk1 = h_k0[qr], h_k1[qr]
    q_wk0, q_wk1 = w_k0[qc], w_k1[qc]
    # A query's ANCHOR extent is its own HR extent divided by the stride: the same
    # region of the picture, s_h*s_w fewer tokens. Dividing (rather than re-tiling
    # the anchor grid) is what `_BlockGridSpec.anchor_h_blocks` does, and it is
    # exact because the contract forces every block edge to be a stride multiple.
    q_ahk0, q_ahk1 = q_hk0 // s_h, q_hk1 // s_h
    q_awk0, q_awk1 = q_wk0 // s_w, q_wk1 // s_w

    # Per-query temporal KV frame range [t0, t1): THE temporal rule, scattered
    # from the shared spec's frame runs onto query rows (the runs partition the
    # query frame axis, so every row is written exactly once).
    t0 = torch.zeros(s_cur, dtype=torch.long, device=dev)
    t1 = torch.zeros(s_cur, dtype=torch.long, device=dev)
    for (qf0, qf1, kf0, kf1) in spec.temporal_runs():
        t0[qf0 * L:qf1 * L] = kf0
        t1[qf0 * L:qf1 * L] = kf1

    win_scope = (anchor_window_scope == "window")
    frame_align = (anchor_align == "frame")

    def mask_mod(b, hh, qi, kvi):
        # HR KV columns live at [A, A + f_kv*L); decode (frame, row, col).
        local = kvi - A
        fkv = local // L
        sp = local % L
        kr = sp // w
        kc = sp % w
        hr_ok = ((fkv >= t0[qi]) & (fkv < t1[qi])
                 & (kr >= q_hk0[qi]) & (kr < q_hk1[qi])
                 & (kc >= q_wk0[qi]) & (kc < q_wk1[qi]))
        if not anchor_given:
            return hr_ok
        # Anchor columns [0, A) on the ANCHOR grid: frame af, spatial (ar, ac).
        af = kvi // L_a
        asp = kvi % L_a
        ar = asp // a_w
        ac = asp % a_w
        if win_scope:
            a_ok = ((ar >= q_ahk0[qi]) & (ar < q_ahk1[qi])
                    & (ac >= q_awk0[qi]) & (ac < q_awk1[qi]))
        else:
            a_ok = torch.ones_like(hr_ok)
        if frame_align:
            a_ok = a_ok & (af == fi[qi])
        return torch.where(kvi < A, a_ok, hr_ok)

    block_mask = create_block_mask(
        mask_mod, B=1, H=None, Q_LEN=s_cur, KV_LEN=A + f_kv * L,
        device=dev, _compile=True)

    if anchor_given:
        k_ext = torch.cat([anchor_k, full_k], dim=0)
        v_ext = torch.cat([anchor_v, full_v], dim=0)
    else:
        k_ext, v_ext = full_k, full_v

    flex = _get_flex_attn()
    out = flex(q_i.transpose(0, 1).unsqueeze(0),
               k_ext.transpose(0, 1).unsqueeze(0),
               v_ext.transpose(0, 1).unsqueeze(0),
               block_mask=block_mask)
    return out.squeeze(0).transpose(0, 1).contiguous()


# ── MagiAttention FFA backend for block-grid windows ─────────────────────────
# Same visibility as the gather/flex branches under the chunk-anchored temporal
# rule (explicit window_radius_t), but executed as flex_flash_attn_func range
# tables over a BLOCK-PACKED sequence. Packing rows into (h-block, w-block,
# frame) groups is what makes the windows cheap: a query block's clamped 3x3
# spatial neighbourhood is a contiguous run of groups and frames are contiguous
# inside a group, so each (neighbour group x temporal span) is ONE rectangle —
# in natural (f, r, c) order the same window strides across rows and explodes
# to per-row granularity.
#
# Correctness rests on one invariant: the q-runs PARTITION the packed q axis.
# FFA reduces rectangles sharing q rows with an online softmax, so a key
# covered by two rectangles of the same q-run is counted twice — in numerator
# AND denominator — which no boolean-mask comparison can see (the MiniMax-H3
# integration shipped that bug; tests below assert multiplicity explicitly).

_MAGI_PLAN_CACHE: dict = {}


def _blockgrid_pack(h, w, bh, bw, F, device):
    """Order rows by (h-block, w-block, frame); natural (r, c) inside a block.

    Returns ``(order, G0, Bs, nb_h, nb_w)``. ``order`` is a device tensor
    mapping packed position -> natural token index for F frames. ``Bs[g]`` and
    ``G0[g]`` are the per-frame token count and packed start of group
    ``g = br*nb_w + bc``, as **Python ints** — the rect table is built from
    them, and one `int()` on a CUDA scalar per rect would mean thousands of
    device->host syncs per geometry (12 012 rects at 34x60 parallel-21f).
    """
    # Block lengths come from the ONE spatial-partition helper (field 1 is the
    # block's own extent; the neighbour radius is irrelevant here).
    h_len = [b[1] for b in _blockgrid_axis_blocks(h, bh, 0)]
    w_len = [b[1] for b in _blockgrid_axis_blocks(w, bw, 0)]
    nb_h, nb_w = len(h_len), len(w_len)
    Bs, G0, start = [], [], 0
    for hl in h_len:                       # g = br*nb_w + bc, row-major
        for wl in w_len:
            Bs.append(hl * wl)
            G0.append(start)               # == F * sum(Bs[:g])
            start += hl * wl * F
    # token (f, r, c) -> group; stable argsort keeps (f, r, c) order inside.
    r = torch.arange(h, device=device).view(1, h, 1)
    c = torch.arange(w, device=device).view(1, 1, w)
    g_hw = ((r // bh) * nb_w + (c // bw)).view(1, h * w)
    g = g_hw.expand(F, h * w).reshape(-1)                      # natural order
    f = torch.arange(F, device=device).repeat_interleave(h * w)
    order = torch.argsort(g * F + f, stable=True)
    return order, G0, Bs, nb_h, nb_w


def _merge_k_spans(spans):
    """Merge touching k spans of ONE q-run into maximal intervals.

    `[a0,a1)` then `[a1,a2)` describe the same visible columns as `[a0,a2)`, and
    in packed order group ``g2``'s rows are `[G0k[g2], G0k[g2]+f_kv*Bk[g2])`, so
    a group's temporal slice touches the next group's start exactly when the
    slice is the whole group (``t0 == 0 and t1 == f_kv``) — the streaming steady
    state, where a block row's neighbour groups collapse into one rectangle
    (1140 -> 780 rects at 30x52, 1716 -> 1170 at 34x60, 2400 -> 1650 at 72x48).
    Under the parallel chunk-causal rule every slice is partial, so nothing
    merges and the table is unchanged.

    A STRICT overlap is not merged but refused: the packing gives each (group,
    run) one interval, so an overlap means two rectangles of one q-run share a
    key — which the FFA online softmax would count twice in numerator AND
    denominator. Absorbing it here would hide the bug the invariant exists for.
    """
    spans = sorted(spans)
    out = [spans[0]]
    for (a0, a1) in spans[1:]:
        cur0, cur1 = out[-1]
        if a0 < cur1:
            raise ValueError(
                f"magi k spans [{cur0},{cur1}) and [{a0},{a1}) overlap within "
                "one q-run — the FFA reduction would count the shared keys "
                "twice; the packing must give each (group, run) one interval.")
        if a0 == cur1:
            out[-1] = (cur0, a1)
        else:
            out.append((a0, a1))
    return out


def _blockgrid_magi_plan(f_cur, f_kv, h, w, bt, block_hw, block_radius_hw,
                         hist, chunk_frames, streaming, anchor, anchor_align,
                         anchor_window_scope, device, *, anchor_hw=None):
    """Range tables for ONE block-grid geometry (cached).

    Builds the packed q permutation, the packed k/v layout [anchor | HR] and
    the rectangle table. Every rectangle is (q0, q1, k0, k1) in PACKED index
    space; q-runs tile the packed q axis exactly once (asserted at build).

    `anchor_hw` is the native LQ-anchor's own, coarser token grid. It gives the
    packed KV TWO layouts — the anchor is packed with block size
    `(bh/s_h, bw/s_w)` on an `(h_a, w_a)` grid, the HR part with `(bh, bw)` on
    `(h, w)` — and shifts the HR base to `A = f_cur*h_a*w_a`. `None` (or `(h, w)`)
    keeps the historical single-pack path bit for bit.
    """
    if anchor_hw is not None and tuple(int(x) for x in anchor_hw) == (h, w):
        anchor_hw = None                     # one grid: the old path, same cache slot
    key = (f_cur, f_kv, h, w, bt, int(block_hw[0]), int(block_hw[1]),
           int(block_radius_hw[0]), int(block_radius_hw[1]), hist,
           None if chunk_frames is None else int(chunk_frames), streaming,
           bool(anchor), anchor_align, anchor_window_scope,
           None if anchor_hw is None else (int(anchor_hw[0]), int(anchor_hw[1])),
           device.type, getattr(device, "index", None))
    cached = _MAGI_PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    bh, bw = int(block_hw[0]), int(block_hw[1])
    rh, rw = int(block_radius_hw[0]), int(block_radius_hw[1])
    L = h * w
    a_h, a_w = (h, w) if anchor_hw is None else (int(anchor_hw[0]),
                                                int(anchor_hw[1]))
    A = f_cur * a_h * a_w if anchor else 0

    order_q, G0q, Bq, nb_h, nb_w = _blockgrid_pack(h, w, bh, bw, f_cur, device)
    order_k, G0k, Bk, _, _ = _blockgrid_pack(h, w, bh, bw, f_kv, device)
    if anchor and anchor_hw is not None:
        # The coarse anchor needs its OWN pack: same picture, `s_h*s_w` fewer
        # tokens, hence block size bh/s_h x bw/s_w on the (a_h, a_w) grid. The
        # block COUNT is unchanged — `s_h | h` and `s_h | bh` force
        # `s_h | (h mod bh)`, so the short tail block divides too — which is what
        # lets the rect builder keep pairing group `g = br*nb_w + bc` across the
        # two packs. Structural, but cheap to verify, and a silent mismatch here
        # would mean anchor spans of the wrong picture region.
        s_h, s_w = h // a_h, w // a_w
        order_a, G0a, Ba, nb_h_a, nb_w_a = _blockgrid_pack(
            a_h, a_w, bh // s_h, bw // s_w, f_cur, device)
        if (nb_h_a, nb_w_a) != (nb_h, nb_w):
            raise ValueError(
                f"anchor grid {(a_h, a_w)} blocked by "
                f"{(bh // s_h, bw // s_w)} gives {nb_h_a}x{nb_w_a} blocks but "
                f"the HR grid {(h, w)} blocked by {(bh, bw)} gives "
                f"{nb_h}x{nb_w}; the magi rect table pairs the two block grids "
                "one to one.")
    else:
        # One grid: the anchor has f_cur frames with the same blocking, so its
        # pack IS the q pack — same permutation, same group table. Reuse it (one
        # argsort fewer, and `order_a is order_q` documents the equality).
        order_a, G0a, Ba = (order_q, G0q, Bq) if anchor else (None, None, None)

    # temporal runs per group: (qf0, qf1, t0, t1) with q frames [qf0, qf1) and
    # HR kv frames [t0, t1) — THE temporal rule, shared with gather/flex.
    runs = _BlockGridSpec(
        f_cur=f_cur, f_kv=f_kv, h=h, w=w, block_t=int(bt),
        bh=bh, bw=bw, rh=rh, rw=rw, hist=int(hist),
        chunk_frames=None if chunk_frames is None else int(chunk_frames),
        streaming=bool(streaming)).temporal_runs()
    if anchor and anchor_align == "frame":   # anchor visibility differs per frame
        runs = [(f0, f0 + 1, t0, t1)
                for (qf0, qf1, t0, t1) in runs
                for f0 in range(qf0, qf1)]
    win_scope = (anchor_window_scope == "window")

    rects = []
    covered = 0
    for br in range(nb_h):
        for bc in range(nb_w):
            g = br * nb_w + bc
            row_lo, row_hi = _blockgrid_neighbours(nb_h, br, rh)
            col_lo, col_hi = _blockgrid_neighbours(nb_w, bc, rw)
            for (qf0, qf1, t0, t1) in runs:
                q0, q1 = G0q[g] + qf0 * Bq[g], G0q[g] + qf1 * Bq[g]
                covered += q1 - q0
                # All k spans of THIS q-run, coalesced once at the end: only
                # per-run spans may be merged (rectangles of different q-runs
                # describe different queries).
                spans = []
                for br2 in range(row_lo, row_hi + 1):
                    for bc2 in range(col_lo, col_hi + 1):
                        g2 = br2 * nb_w + bc2
                        spans.append((A + G0k[g2] + t0 * Bk[g2],
                                      A + G0k[g2] + t1 * Bk[g2]))
                if anchor and anchor_align == "frame":
                    # qf0 == the chunk frame whose LQ anchor this run reads.
                    # Window scope uses the CLAMPED neighborhood extent — the
                    # same 3x3 group set as the HR window (flex/gather patch
                    # anchor tokens inside q_hk0..q_hk1, which at edges is a
                    # shifted window, NOT the query's own block).
                    if win_scope:
                        for br2 in range(row_lo, row_hi + 1):
                            for bc2 in range(col_lo, col_hi + 1):
                                g2 = br2 * nb_w + bc2
                                spans.append((G0a[g2] + qf0 * Ba[g2],
                                              G0a[g2] + (qf0 + 1) * Ba[g2]))
                    else:
                        for g2 in range(nb_h * nb_w):
                            spans.append((G0a[g2] + qf0 * Ba[g2],
                                          G0a[g2] + (qf0 + 1) * Ba[g2]))
                elif anchor:              # chunk-aligned anchor
                    if win_scope:
                        for br2 in range(row_lo, row_hi + 1):
                            for bc2 in range(col_lo, col_hi + 1):
                                g2 = br2 * nb_w + bc2
                                spans.append((G0a[g2],
                                              G0a[g2] + f_cur * Ba[g2]))
                    else:
                        spans.append((0, A))
                rects += [(q0, q1, k0, k1) for (k0, k1)
                          in _merge_k_spans(spans)]
    # NOT an assert: `python -O` would strip it, and a q axis that is not
    # partitioned makes the FFA online softmax silently double-count keys.
    if covered != f_cur * L:
        raise ValueError(
            f"magi q-runs cover {covered} of {f_cur * L} query rows "
            f"(f_cur={f_cur}, h={h}, w={w}, block=({bt},{bh},{bw})) — the FFA "
            "reduction requires the q-runs to partition the packed q axis "
            "exactly once.")

    inv_q = torch.empty_like(order_q)
    inv_q[order_q] = torch.arange(f_cur * L, device=device)
    # ONE host->device transfer for the whole table. `device=` on torch.tensor
    # (and a plain .to()) copies from pageable memory with a stream sync;
    # non_blocking stages it on the stream instead, so the build stays
    # sync-free end to end (pinning it would cost more than it saves for a
    # once-per-geometry table).
    ranges = torch.tensor(rects, dtype=torch.int32).reshape(-1, 4).to(
        device, non_blocking=True)
    plan = {
        "order_q": order_q, "inv_q": inv_q, "order_k": order_k,
        "order_a": order_a, "A": A, "f_kv": f_kv,
        "q_ranges": ranges[:, :2].contiguous(),
        "k_ranges": ranges[:, 2:].contiguous(),
        "n_rects": len(rects),
    }
    _MAGI_PLAN_CACHE[key] = plan
    return plan


def _blockgrid_magi_attend(spec, q_i, full_k, full_v, *,
                           anchor_k=None, anchor_v=None,
                           anchor_align="frame", anchor_window_scope="window"):
    """MagiAttention FFA execution of ONE sample's block-grid windows.

    Packs q and [anchor | KV] into block-major order, runs
    ``flex_flash_attn_func`` over the cached range tables (auto_range_merge on,
    as MAGI-2 ships it) and restores the natural token order. Numerically
    equivalent to the gather/flex branches under the same chunk-anchored rule;
    returns ``[s_cur, n, d]``.
    """
    if _magi_ffa is None:
        raise RuntimeError(
            'attn_impl="magi" needs the magi_attention package '
            "(flex_flash_attn_func); it is not importable in this env.")
    anchor = anchor_k is not None
    plan = _blockgrid_magi_plan(
        spec.f_cur, spec.f_kv, spec.h, spec.w, spec.block_t,
        (spec.bh, spec.bw), (spec.rh, spec.rw), spec.hist, spec.chunk_frames,
        spec.streaming, anchor, anchor_align, anchor_window_scope, q_i.device,
        anchor_hw=spec.anchor_hw)

    qp = q_i.index_select(0, plan["order_q"])
    kp = full_k.index_select(0, plan["order_k"])
    vp = full_v.index_select(0, plan["order_k"])
    if anchor:
        # Host-side, one comparison: an `order_a` built for a different grid is
        # an out-of-bounds index_select, i.e. a device-side assert that poisons
        # the CUDA context for the rest of the process (a training job dies, not
        # just the step). Fail in Python instead.
        if anchor_k.shape[0] != plan["A"]:
            raise ValueError(
                f"magi anchor pack expects {plan['A']} anchor tokens "
                f"(f_cur={spec.f_cur}, anchor grid {spec.anchor_grid}) but got "
                f"{anchor_k.shape[0]}.")
        kp = torch.cat([anchor_k.index_select(0, plan["order_a"]), kp], dim=0)
        vp = torch.cat([anchor_v.index_select(0, plan["order_a"]), vp], dim=0)

    out, _ = _magi_ffa(qp, kp, vp,
                       q_ranges=plan["q_ranges"], k_ranges=plan["k_ranges"],
                       attn_type_map=torch.zeros(plan["n_rects"],
                                                 dtype=torch.int32,
                                                 device=q_i.device),
                       auto_range_merge=True)
    return out.index_select(0, plan["inv_q"]).contiguous()


def mask_free_window_attention(
    q, k, v, grid_sizes,
    pre_cache_k=None,
    pre_cache_v=None,
    kv_len=None,
    anchor_k=None,
    anchor_v=None,
    anchor_align="frame",
    anchor_window_scope="window",
    anchor_hw=None,
    anchor_keep=None,
    attn_impl="auto",
    win_chunk=None,
    causal_chunk_frames=None,
    causal_kv_span=None,
    block_t=None,
    block_hw=(4, 4),
    block_radius_hw=(2, 2),
    block_radius_t=None,
):
    """Mask-free causal block-grid window self-attention with optional KV cache.

    Query blocks hard-partition the (f_cur, h, w) grid into (block_t, bh, bw)
    cuboids; each attends its boundary-clamped (rh, rw) spatial neighborhood over
    a causal temporal span (see ``block_radius_t``). Nothing is masked — the
    visibility IS the index set that gets addressed.

    Args:
        q, k, v: [b, s_cur, n, d], s_cur = f_cur*h*w. q,k are assumed already
            RoPE'd (RoPE is applied globally before windowing, upstream).
        grid_sizes: [b, 3] = (f_cur, h, w) of the CURRENT chunk.
        pre_cache_k, pre_cache_v: [b, f_cache*h*w, n, d] roped history, or None.
        kv_len: number of history latent frames to retain in the returned cache.
        anchor_k, anchor_v: [b, f_cur*h*w, n, d] per-frame LR-anchor K/V, already
            RoPE'd at each frame's absolute temporal offset (streaming path
            only, mutually required). When provided, query frame i attends its
            OWN anchor frame i's K/V prepended to the window KV — strict
            frame-aligned 1:1, matching the dense streaming path. The anchor is
            used for the attention computation ONLY and never enters the KV
            cache. Parallel (cache-free) forwards do not support it.
        anchor_align: "frame" (default) prepends each query frame's OWN anchor
            frame (strict 1:1, matches the dense per-frame path); "chunk"
            prepends the ENTIRE anchor chunk to every query (matches the
            per-chunk training/inference semantics, e.g. Causal-Forcing-VSR,
            where any HR frame can attend any LR anchor frame of the chunk).
        anchor_window_scope: SPATIAL extent of the anchor inside the window path.
            "window" (default) restricts the anchor to the query block's own
            spatial index set — its clamped neighborhood extent, the same span as
            its HR KV — so the anchor costs ``khL*kwL`` tokens per anchor frame
            instead of a whole frame. "global" keeps the pre-existing behaviour
            where every window attends the ENTIRE anchor frame/chunk; kept as an
            A-B escape hatch. Only the window path honours this: the dense paths
            are spatially global on the HR side too, so a full anchor is
            consistent there.
        anchor_hw: the anchor's OWN token grid ``(h_a, w_a)`` when it is coarser
            than the query grid — the NATIVE LQ-anchor, where the anchor is a
            shrunken copy of the target rather than an upsampled one, so it
            carries ``f_cur*h_a*w_a`` tokens instead of ``f_cur*h*w``. Each query
            block then attends the SAME picture extent at the coarse resolution:
            its HR neighbourhood ``(kh0..kh1, kw0..kw1)`` divided by the stride
            ``(h//h_a, w//w_a)``, which must divide both the grid and the block
            size. ``None`` (default) — and ``(h, w)``, which normalises to
            ``None`` — is the pre-existing single-grid path, bit for bit.
            Inference only: ``attn_impl="flex"``/``"magi"`` reject it.
        anchor_keep: optional per-sample gate, bool [b] (or any indexable of
            length b). ``False`` means that sample attends to NO anchor — the
            unified cond_drop semantics of the dense paths. This is structural
            here (the anchor columns are simply not gathered) rather than a
            masked/varlen trick, so a dropped sample is bit-identical to the
            same call with ``anchor_k=anchor_v=None``. ``None`` keeps every
            sample's anchor (inference / no drop).
        attn_impl: "auto" | "flash" | "sdpa" select the gather backend's kernel —
            per-window KV copies + a batched dense attention. Autograd-safe, but
            the O(Nw*Lkv) gathered KV stays live for the backward, so it is the
            memory-hungry choice for training. "flex": flex_attention + BlockMask over
            the flat [anchor | KV] sequence — numerically equivalent, but no
            per-window KV gather is materialised, so training backward holds a
            single K/V copy (memory lever for DMD rollouts). "magi" (requires
            block_radius_t): MagiAttention flex_flash_attn_func over a
            block-packed range table; same visibility as the others under the
            chunk-anchored temporal rule. "triton": fused Triton kernels that
            gather nothing at all — one program per (window, m-tile, head)
            streams its own KV cuboid with an online softmax, LQ anchor
            included, so no per-window KV copy and no block-major pack/unpack
            happens. The fastest backend for INFERENCE, and forward-only: it
            raises under autograd, and it requires
            ``anchor_window_scope="window"``.
        win_chunk: process windows in batches of this many to bound peak memory.
            None / >= Nw runs all windows at once (fastest, highest peak mem).
            Numerically identical to the one-shot path regardless of the value.
        causal_chunk_frames, causal_kv_span: enable the chunk-causal parallel path
            (both in latent frames). Only used when there is NO KV cache, i.e. a
            full-sequence forward, where an unconstrained temporal span would
            otherwise let queries attend to future frames. With a cache, causality
            is already guaranteed structurally (only past frames are held) and
            ``causal_kv_span`` is ignored. ``causal_kv_span`` counts the query's
            own chunk, so matching a streaming cache of ``kv_len`` frames means
            ``causal_kv_span = kv_len + causal_chunk_frames``.
        block_t, block_hw, block_radius_hw: query-block size along time and space,
            and the spatial neighborhood radius in block units. ``block_t``
            defaults to ``causal_chunk_frames``.
        block_radius_t: explicit TEMPORAL radius in block_t units. When set, the
            temporal rule becomes chunk-anchored: every query attends its whole
            own chunk (causal_chunk_frames, or the whole current chunk when
            streaming) plus the last ``(2*block_radius_t+1)*block_t`` history
            frames. Streaming and parallel receptive fields are then
            frame-identical by construction. Required for attn_impl="magi".
            When left None, the legacy rule derives the temporal radius from the
            span the op already receives (``causal_kv_span//block_t - 1``
            parallel, ``f_kv//block_t - 1`` streaming).

    Returns:
        out: [b, s_cur, n, d]
        new_cache_k, new_cache_v: [b, kept*h*w, n, d] (token layout, self-managed).
    """
    b, s_cur, n, d = q.shape
    impl = str(attn_impl)
    if impl not in BLOCKGRID_IMPLS:
        # Fail fast instead of degrading to gather: a typo'd or not-yet-wired
        # backend must not quietly produce a correct number under the wrong name.
        raise ValueError(
            f"attn_impl must be one of {BLOCKGRID_IMPLS}, got {impl!r}.")
    outs, new_ck, new_cv = [], [], []
    anchor_hw_arg = None if anchor_hw is None else (int(anchor_hw[0]),
                                                   int(anchor_hw[1]))
    for i in range(b):
        f_cur, h, w = (int(x) for x in grid_sizes[i].tolist())
        assert f_cur * h * w == s_cur, f"grid {(f_cur, h, w)} inconsistent with s_cur={s_cur}"
        q_i, k_i, v_i = q[i], k[i], v[i]                      # [s_cur, n, d]

        if pre_cache_k is not None and pre_cache_v is not None:
            full_k = torch.cat([pre_cache_k[i], k_i], dim=0)
            full_v = torch.cat([pre_cache_v[i], v_i], dim=0)
        else:
            full_k, full_v = k_i, v_i
        s_kv = full_k.shape[0]
        f_kv = s_kv // (h * w)
        assert f_kv * h * w == s_kv, "KV cache is not aligned to whole frames"

        # `anchor_given` is batch-level (an anchor tensor was passed at all);
        # `use_anchor` is the per-sample decision after the cond_drop gate. The
        # two are kept apart on purpose: shape/mode validation and the streaming
        # regime detection below must not flip just because THIS sample's anchor
        # was dropped.
        anchor_given = (anchor_k is not None) or (anchor_v is not None)
        use_anchor = anchor_given
        # An anchor grid EQUAL to the query grid is not a second geometry — it is
        # the old path. Normalising it to None here is what keeps every cache key,
        # plan key and index row bit-identical for the (h, w) case.
        a_hw = (None if anchor_hw_arg is None or anchor_hw_arg == (h, w)
                else anchor_hw_arg)
        if anchor_given:
            if anchor_k is None or anchor_v is None:
                raise ValueError("mask_free_window_attention: anchor_k and "
                                 "anchor_v must be provided together.")
            if (causal_chunk_frames is not None and causal_kv_span is not None
                    and pre_cache_k is None and pre_cache_v is None):
                # The parallel chunk-causal windows use query-time blocks over
                # the full grid, which the per-frame anchor loop cannot express
                # yet. The streaming path (cache present) and the plain
                # bidirectional forward (no cache, e.g. the first chunk of the
                # causal sf pipeline, where f_kv == f_cur) are both supported.
                raise NotImplementedError(
                    "LQ-anchor in the parallel chunk-causal window path is not "
                    "supported; anchor injection requires the streaming path "
                    "(KV cache present) or a plain bidirectional window forward "
                    "(no KV cache and no causal_chunk_frames/causal_kv_span).")
            # Validates the divisibility contract too, so an illegal grid raises
            # here rather than producing a ragged plan downstream.
            s_h, s_w = _anchor_grid_stride(h, w, int(block_hw[0]),
                                           int(block_hw[1]), a_hw)
            L = (h // s_h) * (w // s_w)
            if anchor_align not in ("frame", "chunk"):
                raise ValueError(
                    f"anchor_align must be 'frame' or 'chunk', got {anchor_align!r}.")
            if anchor_window_scope not in ("window", "global"):
                raise ValueError(
                    f"anchor_window_scope must be 'window' or 'global', got "
                    f"{anchor_window_scope!r}.")
            if anchor_k.shape[1] != f_cur * L:
                grid = "h*w" if a_hw is None else f"h_a*w_a ({a_hw})"
                raise ValueError(
                    f"frame-aligned LQ-anchor needs f_cur*{grid} = {f_cur * L} "
                    f"tokens per sample, got anchor_k sequence length "
                    f"{anchor_k.shape[1]} (anchor must be harvested at the "
                    f"resolution anchor_hw declares).")
            if anchor_keep is not None:
                keep_i = (anchor_keep.view(-1)[i] if torch.is_tensor(anchor_keep)
                          else anchor_keep[i])
                use_anchor = bool(keep_i)

        # Causal block-grid (MAGI-style) neighborhood windows. Query blocks
        # hard-partition the grid; each attends the causal neighborhood
        # cuboid. `rt` (temporal radius) is derived from the causal span the
        # op already receives (parallel) or from the cache length (stream):
        # both give the same frame-for-frame receptive field. An explicit
        # `block_radius_t` switches to the chunk-anchored rule instead
        # (own chunk + (2r+1)*block_t history frames) — the b/r
        # parametrization the magi backend is built for.
        bt = int(block_t if block_t is not None else causal_chunk_frames)
        if bt <= 0:
            raise ValueError(f"block_t must be positive, got {bt}.")
        # f_cur % bt is only required when the temporal axis is PARTITIONED
        # into bt-sized query blocks (parallel multi-block forward, or a
        # streaming chunk of >= bt frames). A sub-block streaming forward
        # (f_cur < bt, no anchor) is a single query block — e.g. the 1-frame
        # LQ-anchor harvest.
        is_streaming = (pre_cache_k is not None or pre_cache_v is not None
                        or anchor_given)
        hist = (None if block_radius_t is None
                else (2 * int(block_radius_t) + 1) * bt)
        if hist is None:
            if not is_streaming and f_cur % bt != 0 and f_cur >= bt:
                raise ValueError(
                    f"block-grid window attention needs f_cur ({f_cur}) to be a "
                    f"multiple of block_t ({bt}).")
        elif not is_streaming and causal_chunk_frames:
            cf = int(causal_chunk_frames)
            if f_cur % cf != 0 and f_cur >= cf:
                raise ValueError(
                    f"chunk-anchored block-grid windows need f_cur ({f_cur}) "
                    f"to be a multiple of causal_chunk_frames ({cf}).")
        if causal_kv_span is not None:
            rt = int(causal_kv_span) // bt - 1
        else:
            rt = f_kv // bt - 1
        # Sub-block KV grid (e.g. 1-frame harvest): degenerate to attending
        # the available frames instead of an empty span.
        rt = max(rt, 0)

        # ONE geometry for all three backends: they differ in execution only.
        spec = _BlockGridSpec(
            f_cur=f_cur, f_kv=f_kv, h=h, w=w, block_t=bt,
            bh=int(block_hw[0]), bw=int(block_hw[1]),
            rh=int(block_radius_hw[0]), rw=int(block_radius_hw[1]),
            hist=hist, rt=rt,
            chunk_frames=(None if causal_chunk_frames is None
                          else int(causal_chunk_frames)),
            streaming=is_streaming,
            anchor_hw=(a_hw if anchor_given else None))
        # The LQ-anchor is prepended to the KV for the attention only (never
        # cached); `use_anchor` is this sample's post-cond_drop decision, so a
        # dropped sample is bit-identical to anchor_k=anchor_v=None.
        anchor_kw = dict(
            anchor_k=(anchor_k[i] if use_anchor else None),
            anchor_v=(anchor_v[i] if use_anchor else None),
            anchor_align=anchor_align, anchor_window_scope=anchor_window_scope)

        if impl == "magi":
            if hist is None:
                raise ValueError(
                    'attn_impl="magi" needs an explicit block_radius_t '
                    "(the chunk-anchored temporal rule).")
            out_i = _blockgrid_magi_attend(spec, q_i, full_k, full_v, **anchor_kw)
        elif impl == "flex":
            out_i = _blockgrid_flex_attend(spec, q_i, full_k, full_v, **anchor_kw)
        elif impl == "triton":
            out_i = _blockgrid_triton_attend(spec, q_i, full_k, full_v,
                                             **anchor_kw)
        else:
            out_i = _blockgrid_gather_attend(
                spec, q_i, full_k, full_v, attn_impl=impl,
                win_chunk=win_chunk, **anchor_kw)

        outs.append(out_i)
        ck_i, cv_i = _truncate_kv_cache(full_k, full_v, kv_len, h * w, f_kv)
        new_ck.append(ck_i)
        new_cv.append(cv_i)

    out = torch.stack(outs, dim=0)
    if b == 1:
        return out, new_ck[0].unsqueeze(0), new_cv[0].unsqueeze(0)
    return out, torch.stack(new_ck, dim=0), torch.stack(new_cv, dim=0)
