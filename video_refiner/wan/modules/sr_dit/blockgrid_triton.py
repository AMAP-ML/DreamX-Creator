# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Fused Triton kernels for block-grid (MAGI-style) window self-attention.

The fourth backend of ``attention.mask_free_window_attention`` (``attn_impl=
"triton"``). Where the gather backend materialises every window's KV cuboid with
``index_select`` and scatters the result back, and the magi backend packs the
sequence into block-major order to express the windows as FFA rectangles, these
kernels do neither: one program per (window, m-tile, head) loads its own Q rows
and streams its KV cuboid straight from the grid with an online softmax, storing
its output tokens in place. Nothing is gathered, packed or unpacked, and the LQ
anchor is fused into the same launch.

Every block is covered — edge and tail included — so there is no hybrid with the
gather path.

**One JIT compile per kernel per process.** Every extent (query block origin and
size, KV cuboid, anchor frame count) rides in a per-window int32 params row read
at runtime; only ``(D, BLOCK_M, BLOCK_N)`` are ``constexpr``. Specializing the
kernel per shape instead costs 36-144 compiles per run, which is enough to lose
the per-call win end to end. Triton's own scalar specialization (``% 16 == 0`` /
``== 1``) can still split a kernel by the divisibility of ``H``/``W``/``HR_BASE``
and by ``M_TILES == 1``, i.e. one compile per DiT grid and one extra for a
1-frame forward; nothing scales with the streaming ``f_kv`` ramp or the ragged
tail blocks. See :func:`triton_compile_keys` and ``attention.
prewarm_blockgrid_triton``.

The kernels measure 1.47x/1.51x the magi backend at the two production grids,
with bf16-ulp-identical outputs. :func:`build_plan` reads ``_BlockGridSpec``
(``temporal_runs`` / ``h_blocks`` / ``w_blocks``), which keeps all four backends
on one definition of visibility — including the chunk-anchored temporal rule.

Forward only: the kernels write into a raw output buffer and record nothing for
autograd. Training uses ``attn_impl="flex"`` or ``"magi"``.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ModuleNotFoundError:                      # keep the module importable
    tl = None
    TRITON_AVAILABLE = False

    class _NoTriton:
        """Stands in for the ``triton`` module so ``@triton.jit`` still imports.

        The ``D: tl.constexpr`` annotations are lazy (``from __future__ import
        annotations``), so an undecorated function body never evaluates ``tl``.
        """

        @staticmethod
        def jit(fn):
            return fn

    triton = _NoTriton()


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


# Params row layouts (int32, one row per window), shared by the planner and the
# kernels: the kernel reads them with raw pointer arithmetic, so the two must
# not drift. All lengths are LENGTHS, not end indices.
#
# The last six fields are the ANCHOR's own spatial geometry (``_BlockGridSpec.
# anchor_hw``, the native LQ-anchor): its neighbourhood extent and grid, i.e. the
# HR extent divided by the stride. With ``anchor_hw=None`` they hold the HR values
# verbatim, so ``L_a == L`` / ``sp_a == sp`` and the kernels do exactly the
# arithmetic they did before the second geometry existed.
_P_FRAME = 17   # q_h0 q_w0 qhL qwL kv_t0 kv_tL kh0 khL kw0 kwL f_cur
                # + kh0_a khL_a kw0_a kwL_a H_a W_a
_P_PLAIN = 19   # q_t0 q_tL q_h0 q_w0 qhL qwL kv_t0 kv_tL kh0 khL kw0 kwL a_frames
                # + kh0_a khL_a kw0_a kwL_a H_a W_a


def build_plan(spec, *, anchor_mode=None, block_n=64):
    """ONE launch spec covering every window of ``spec``: [Nw, P] int32 + tiles.

    ``spec`` is duck-typed (``_BlockGridSpec``): the geometry is read ONLY through
    ``spec.temporal_runs()``, ``spec.h_blocks()`` and ``spec.w_blocks()``, so the
    triton backend cannot drift from the other three.

    ``anchor_mode``:
      * ``"frame"`` — per-frame LQ anchor, the frame kernel. Its Q spans the whole
        chunk and the row carries no ``q_t0``, so the query time axis must be a
        single run ``[0, f_cur)``. An anchor implies ``streaming`` implies
        ``spec.one_query_run``, so that holds by construction; it is checked
        rather than assumed because a monkeypatched or future rule could break it.
      * ``"chunk"`` / ``None`` — the plain kernel, one row per (temporal run,
        h-block, w-block), with the whole anchor chunk visible (``a_frames =
        f_cur``) or no anchor at all (``a_frames = 0``).

    ``block_m`` is sized by the LARGEST window and shared by all of them (the
    launch grid is uniform in m-tile); shorter windows skip their tail tiles.

    The anchor's spatial geometry is read through ``anchor_h_blocks()`` /
    ``anchor_w_blocks()`` / ``anchor_grid`` (the native LQ-anchor's coarser grid);
    a spec that predates those falls back to the HR blocks, which is the
    degenerate — and pre-existing — case. ``plan["hr_base"]`` is the offset of the
    HR KV inside ``cat([anchor, HR])``, i.e. ``f_cur * h_a * w_a``: anchor tokens,
    which are HR tokens only when the grids coincide.
    """
    hb, wb = spec.h_blocks(), spec.w_blocks()
    hb_a = spec.anchor_h_blocks() if hasattr(spec, "anchor_h_blocks") else hb
    wb_a = spec.anchor_w_blocks() if hasattr(spec, "anchor_w_blocks") else wb
    a_h, a_w = getattr(spec, "anchor_grid", (int(spec.h), int(spec.w)))
    runs = spec.temporal_runs()
    f_cur = int(spec.f_cur)
    rows, lqs = [], []
    if anchor_mode == "frame":
        if len(runs) != 1 or runs[0][0] != 0 or runs[0][1] != f_cur:
            raise ValueError(
                f'anchor_mode="frame" needs the query time axis to be a single '
                f"run [0, {f_cur}), got {runs}. The per-frame anchor kernel has "
                f"no q_t0 field; use anchor_mode=None/'chunk' for a partitioned "
                f"query axis.")
        _, _, kv0, kv1 = runs[0]
        for hi, (qh0, qhL, kh0, khL) in enumerate(hb):
            for wi, (qw0, qwL, kw0, kwL) in enumerate(wb):
                rows.append([qh0, qw0, qhL, qwL, kv0, kv1 - kv0,
                             kh0, khL, kw0, kwL, f_cur,
                             hb_a[hi][2], hb_a[hi][3], wb_a[wi][2], wb_a[wi][3],
                             a_h, a_w])
                lqs.append(f_cur * qhL * qwL)
    else:
        a_frames = f_cur if anchor_mode == "chunk" else 0
        for (q0, q1, kv0, kv1) in runs:
            for hi, (qh0, qhL, kh0, khL) in enumerate(hb):
                for wi, (qw0, qwL, kw0, kwL) in enumerate(wb):
                    rows.append([q0, q1 - q0, qh0, qw0, qhL, qwL,
                                 kv0, kv1 - kv0, kh0, khL, kw0, kwL, a_frames,
                                 hb_a[hi][2], hb_a[hi][3],
                                 wb_a[wi][2], wb_a[wi][3], a_h, a_w])
                    lqs.append((q1 - q0) * qhL * qwL)
    # >=16 so tl.dot keeps a sane M even on tiny test grids; <=64 to bound
    # registers (Lq is 192 at the shipped geometry -> 3 tiles of 64).
    block_m = max(16, min(_next_pow2(max(lqs)), 64))
    m_tiles = max((x + block_m - 1) // block_m for x in lqs)
    return {"params": torch.tensor(rows, dtype=torch.int32).contiguous(),
            "n_w": len(rows), "block_m": block_m, "m_tiles": m_tiles,
            "block_n": block_n,
            "hr_base": (f_cur * a_h * a_w) if anchor_mode is not None else 0,
            "p_stride": _P_FRAME if anchor_mode == "frame" else _P_PLAIN}


# Keyed on the geometry, NOT on the temporal rule the geometry implies — a test
# that swaps `temporal_runs` must clear this (attention._clear_window_caches).
TRITON_PLAN_CACHE: dict = {}


def cached_plan(spec, *, anchor_mode=None, block_n=64):
    """:func:`build_plan` memoized. The plan is pure host-side integer work but
    would otherwise be rebuilt ~360 times per clip (30 layers x forwards), each
    time with a host->device copy of the params."""
    key = (tuple(spec), anchor_mode, block_n)
    plan = TRITON_PLAN_CACHE.get(key)
    if plan is None:
        plan = TRITON_PLAN_CACHE[key] = build_plan(
            spec, anchor_mode=anchor_mode, block_n=block_n)
    return plan


def triton_compile_keys(plan, *, anchor_mode=None, d=None):
    """The kernel specializations this plan needs — one, by construction.

    Only the constexprs select a compile, and ``block_m`` is pinned by the
    largest window (64 at any geometry whose blocks hold >= 64 query rows, i.e.
    the shipped 8x8 x chunk 3). Grid size, ragged tails and the streaming cache
    length do not appear. Triton's own scalar specialization can still split a
    kernel further, so this is a lower bound used by the tests as a regression
    backstop, not an oracle.
    """
    return {("frame" if anchor_mode == "frame" else "plain",
             plan["block_m"], plan["block_n"], d)}


@triton.jit
def _blockgrid_fused_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, p_ptr,
    scale, M_TILES, HR_BASE, H, W, TOK_STRIDE, P_STRIDE,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """No-anchor / anchor-"chunk" windows. KV = anchor prefix + HR cuboid."""
    pid0 = tl.program_id(0)
    pid_h = tl.program_id(1)
    wid = pid0 // M_TILES
    m_tile = pid0 % M_TILES
    p = p_ptr + wid * P_STRIDE
    q_t0 = tl.load(p + 0)
    q_tl = tl.load(p + 1)
    q_h0 = tl.load(p + 2)
    q_w0 = tl.load(p + 3)
    qh_l = tl.load(p + 4)
    qw_l = tl.load(p + 5)
    kv_t0 = tl.load(p + 6)
    kv_tl = tl.load(p + 7)
    kh0 = tl.load(p + 8)
    kh_l = tl.load(p + 9)
    kw0 = tl.load(p + 10)
    kw_l = tl.load(p + 11)
    a_frames = tl.load(p + 12)
    # the anchor's own spatial geometry (== the HR one unless anchor_hw shrinks it)
    kh0_a = tl.load(p + 13)
    kh_l_a = tl.load(p + 14)
    kw0_a = tl.load(p + 15)
    kw_l_a = tl.load(p + 16)
    h_a = tl.load(p + 17)
    w_a = tl.load(p + 18)

    q_sp = qh_l * qw_l
    lq = q_tl * q_sp
    m0 = m_tile * BLOCK_M
    if m0 < lq:                      # windows shorter than M_TILES skip the tail
        L = H * W
        sp = kh_l * kw_l
        m = m0 + tl.arange(0, BLOCK_M)
        mask_m = m < lq
        t = m // q_sp
        r = m % q_sp
        hh = r // qw_l
        ww = r % qw_l
        q_flat = (q_t0 + t) * L + (q_h0 + hh) * W + (q_w0 + ww)
        dims = tl.arange(0, D)
        q = tl.load(q_ptr + q_flat[:, None] * TOK_STRIDE + pid_h * D
                    + dims[None, :], mask=mask_m[:, None], other=0.0)

        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # The anchor lives on ITS OWN grid: sp_a tokens per frame, not sp.
        L_a = h_a * w_a
        sp_a = kh_l_a * kw_l_a
        a_len = a_frames * sp_a
        n_kv = a_len + kv_tl * sp
        for i in range(0, (n_kv + BLOCK_N - 1) // BLOCK_N):
            j = i * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_j = j < n_kv
            # anchor prefix (unmasked): frame = j // sp_a, cuboid within the frame
            ja = j // sp_a
            ra = j % sp_a
            flat_a = (ja * L_a + (kh0_a + ra // kw_l_a) * w_a
                      + (kw0_a + ra % kw_l_a))
            # HR part. jh < 0 on the anchor lanes: the indices are computed but
            # discarded by the tl.where below, so nothing out of range is read.
            jh = j - a_len
            rh_ = jh % sp
            flat_h = (HR_BASE + (kv_t0 + jh // sp) * L
                      + (kh0 + rh_ // kw_l) * W + (kw0 + rh_ % kw_l))
            flat = tl.where(j < a_len, flat_a, flat_h)

            k = tl.load(k_ptr + flat[:, None] * TOK_STRIDE + pid_h * D
                        + dims[None, :], mask=mask_j[:, None], other=0.0)
            v = tl.load(v_ptr + flat[:, None] * TOK_STRIDE + pid_h * D
                        + dims[None, :], mask=mask_j[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * scale
            qk = tl.where(mask_j[None, :], qk, float("-inf"))
            # rows whose tile is fully masked must not poison the running stats
            # with -inf - -inf; keep them unchanged.
            row_max = tl.max(qk, 1)
            valid_rows = row_max > float("-inf")
            m_new = tl.where(valid_rows, tl.maximum(m_i, row_max), m_i)
            alpha = tl.where(valid_rows, tl.exp(m_i - m_new), 1.0)
            pr = tl.where(valid_rows[:, None], tl.exp(qk - m_new[:, None]), 0.0)
            l_i = alpha * l_i + tl.sum(pr, 1)
            acc = acc * alpha[:, None] + tl.dot(pr.to(k.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = o_ptr + q_flat[:, None] * TOK_STRIDE + pid_h * D + dims[None, :]
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _blockgrid_fused_frame_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, p_ptr,
    scale, M_TILES, HR_BASE, H, W, TOK_STRIDE, P_STRIDE,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Per-frame anchor: query frame t attends ONLY anchor segment t, then the
    HR cuboid shared by the whole chunk. All f_cur frames in one launch."""
    pid0 = tl.program_id(0)
    pid_h = tl.program_id(1)
    wid = pid0 // M_TILES
    m_tile = pid0 % M_TILES
    p = p_ptr + wid * P_STRIDE
    q_h0 = tl.load(p + 0)
    q_w0 = tl.load(p + 1)
    qh_l = tl.load(p + 2)
    qw_l = tl.load(p + 3)
    kv_t0 = tl.load(p + 4)
    kv_tl = tl.load(p + 5)
    kh0 = tl.load(p + 6)
    kh_l = tl.load(p + 7)
    kw0 = tl.load(p + 8)
    kw_l = tl.load(p + 9)
    f_cur = tl.load(p + 10)
    # the anchor's own spatial geometry (== the HR one unless anchor_hw shrinks it)
    kh0_a = tl.load(p + 11)
    kh_l_a = tl.load(p + 12)
    kw0_a = tl.load(p + 13)
    kw_l_a = tl.load(p + 14)
    h_a = tl.load(p + 15)
    w_a = tl.load(p + 16)

    q_sp = qh_l * qw_l
    lq = f_cur * q_sp
    m0 = m_tile * BLOCK_M
    if m0 < lq:
        L = H * W
        sp = kh_l * kw_l
        m = m0 + tl.arange(0, BLOCK_M)
        mask_m = m < lq
        t = m // q_sp
        r = m % q_sp
        hh = r // qw_l
        ww = r % qw_l
        q_flat = t * L + (q_h0 + hh) * W + (q_w0 + ww)
        dims = tl.arange(0, D)
        q = tl.load(q_ptr + q_flat[:, None] * TOK_STRIDE + pid_h * D
                    + dims[None, :], mask=mask_m[:, None], other=0.0)

        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # One masked anchor segment per frame, but only over the frames THIS
        # m-tile actually holds: at the shipped geometry Lq_frame == BLOCK_M, so
        # that is a single segment instead of f_cur (the rest would be fully
        # masked work).
        L_a = h_a * w_a
        sp_a = kh_l_a * kw_l_a
        n_tiles_a = (sp_a + BLOCK_N - 1) // BLOCK_N
        for s in range(m0 // q_sp, (tl.minimum(lq, m0 + BLOCK_M) - 1) // q_sp + 1):
            seg = (t == s)[:, None]
            for i in range(0, n_tiles_a):
                j = i * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_j = j < sp_a
                flat_a = (s * L_a + (kh0_a + j // kw_l_a) * w_a
                          + (kw0_a + j % kw_l_a))
                k = tl.load(k_ptr + flat_a[:, None] * TOK_STRIDE + pid_h * D
                            + dims[None, :], mask=mask_j[:, None], other=0.0)
                v = tl.load(v_ptr + flat_a[:, None] * TOK_STRIDE + pid_h * D
                            + dims[None, :], mask=mask_j[:, None], other=0.0)
                qk = tl.dot(q, tl.trans(k)) * scale
                qk = tl.where(mask_j[None, :] & seg, qk, float("-inf"))
                row_max = tl.max(qk, 1)
                valid_rows = row_max > float("-inf")
                m_new = tl.where(valid_rows, tl.maximum(m_i, row_max), m_i)
                alpha = tl.where(valid_rows, tl.exp(m_i - m_new), 1.0)
                pr = tl.where(valid_rows[:, None],
                              tl.exp(qk - m_new[:, None]), 0.0)
                l_i = alpha * l_i + tl.sum(pr, 1)
                acc = acc * alpha[:, None] + tl.dot(pr.to(k.dtype), v)
                m_i = m_new

        # shared HR causal cuboid (unmasked)
        n_kv = kv_tl * sp
        for i in range(0, (n_kv + BLOCK_N - 1) // BLOCK_N):
            j = i * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_j = j < n_kv
            rh_ = j % sp
            flat_h = (HR_BASE + (kv_t0 + j // sp) * L
                      + (kh0 + rh_ // kw_l) * W + (kw0 + rh_ % kw_l))
            k = tl.load(k_ptr + flat_h[:, None] * TOK_STRIDE + pid_h * D
                        + dims[None, :], mask=mask_j[:, None], other=0.0)
            v = tl.load(v_ptr + flat_h[:, None] * TOK_STRIDE + pid_h * D
                        + dims[None, :], mask=mask_j[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * scale
            qk = tl.where(mask_j[None, :], qk, float("-inf"))
            row_max = tl.max(qk, 1)
            valid_rows = row_max > float("-inf")
            m_new = tl.where(valid_rows, tl.maximum(m_i, row_max), m_i)
            alpha = tl.where(valid_rows, tl.exp(m_i - m_new), 1.0)
            pr = tl.where(valid_rows[:, None], tl.exp(qk - m_new[:, None]), 0.0)
            l_i = alpha * l_i + tl.sum(pr, 1)
            acc = acc * alpha[:, None] + tl.dot(pr.to(k.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = o_ptr + q_flat[:, None] * TOK_STRIDE + pid_h * D + dims[None, :]
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None])


def run_blockgrid_triton(q, k_ext, v_ext, out, *, plan, f_cur, h, w, n, d,
                        anchor_mode=None):
    """Launch the fused path for one sample. Writes into ``out``, returns it.

    ``q``/``out``: [f_cur*L, n, d]. ``k_ext``/``v_ext``: the extended KV the
    gather backend also builds — ``cat([anchor, HR])`` when an anchor is present
    (so ``HR_BASE = f_cur*h*w``), otherwise the HR KV alone (``HR_BASE = 0``).
    All must be contiguous with a stride-1 head dim.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("attn_impl='triton' needs the triton package.")
    params = plan.get(("dev", q.device))
    if params is None:
        # Re-read by every launch and never changed: the H2D copy belongs
        # outside the hot path, cached per (geometry, device).
        params = plan[("dev", q.device)] = plan["params"].to(q.device)
    mt = plan["m_tiles"]
    kernel = (_blockgrid_fused_frame_kernel if anchor_mode == "frame"
              else _blockgrid_fused_kernel)
    # The HR KV starts after the anchor, which is f_cur*h_a*w_a tokens long —
    # h*w only when the anchor shares the HR grid. `.get` keeps hand-built plans
    # (older tests) working on the pre-`hr_base` expression.
    hr_base = plan.get("hr_base",
                       f_cur * h * w if anchor_mode is not None else 0)
    kernel[(plan["n_w"] * mt, n)](
        q, k_ext, v_ext, out, params,
        d ** -0.5, mt, hr_base,
        h, w, n * d, plan["p_stride"],
        D=d, BLOCK_M=plan["block_m"], BLOCK_N=plan["block_n"], num_warps=4,
    )
    return out
