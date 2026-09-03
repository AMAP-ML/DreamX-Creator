from typing import List, Optional
import contextlib
import logging
import time
import torch
import torch.nn.functional as F

from utils.sr_dit_wrapper import SRDiTWrapper, WanVAEWrapper22
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
from wan.modules.sr_dit.attention import prewarm_blockgrid_triton

import tqdm


class StageTimer:
    """Coarse wall-clock accounting for one SR clip: ``pre`` / ``dit`` / ``decode``.

    Why a split and not just the aggregate ``SR done: ...``: the speed knobs land
    on *different* stages. ``--window_attn_impl`` and ``--fp8_linear`` only touch
    ``dit``; the student decoder (``--enable_nu_lightvae``) only touches
    ``decode``. The aggregate cannot distinguish "a knob worked" from "a knob
    did nothing"; the split can.

    Always on, because a number you have to remember to enable is a number you do
    not have for the run you already did. The cost is ~6 ``cudaDeviceSynchronize``
    per clip (per-block detail, which syncs 11x more, stays behind ``PROFILE_SR=1``).
    Synchronising is not optional bookkeeping: launches are async, so without it a
    stage ends while its kernels are still in flight and whichever later stage first
    blocks is billed for them — which would move nearly all of ``dit`` into
    ``decode``, i.e. destroy exactly the split this exists to measure.

    Two recording shapes, because the call sites have two shapes:

      * ``stage(name)`` — context manager, for a single wrapped call (the decode).
      * ``tick()`` / ``lap(name, t)`` — sequential marks, for regions spanning a
        hundred-plus lines (the setup/prefill, the denoise loop). Marking avoids
        re-indenting the whole loop body under a ``with``, and consecutive laps are
        contiguous by construction, so no time falls between two stages unseen.
    """

    #: Report order — the timeline, not the alphabet (which would lead with decode).
    ORDER = ("pre", "dit", "decode")

    def __init__(self, sync=None, detail=False):
        self._sync = sync if sync is not None else (lambda: None)
        #: Per-block DiT times. One line per block per clip is unreadable across a
        #: 30-clip bench, so it is opt-in via ``PROFILE_SR=1``.
        self.detail = bool(detail)
        self.reset()

    def reset(self):
        """Start a new clip. Without this the numbers become a running total over
        the whole bench — still plausible-looking, just monotonically wrong."""
        self.totals = {}
        self.blocks = []

    def tick(self):
        self._sync()
        return time.perf_counter()

    def lap(self, name, t0):
        """Record ``now - t0`` under ``name`` and return the new mark."""
        now = self.tick()
        self.add(name, now - t0)
        return now

    def add(self, name, seconds):
        self.totals[name] = self.totals.get(name, 0.0) + float(seconds)

    @contextlib.contextmanager
    def stage(self, name):
        t0 = self.tick()
        try:
            yield
        finally:
            self.lap(name, t0)

    def report(self, total_s=None):
        """One-line breakdown. ``total_s`` is the caller's own end-to-end number
        (``time.time()`` around the whole clip), which is what the stages have to
        add up to."""
        named = [k for k in self.ORDER if k in self.totals]
        named += sorted(k for k in self.totals if k not in self.ORDER)
        items = [(k, self.totals[k]) for k in named]
        if total_s is None:
            total = sum(self.totals.values())
        else:
            total = float(total_s)
            # `other` = measured by the caller, claimed by no stage. Deliberately
            # signed: negative means a stage is double-counted (nested laps on one
            # name), and clamping to zero would hide that bug behind a tidy 0.0s.
            items.append(("other", total - sum(self.totals.values())))
        parts = [f"{k}={v:.1f}s {(100.0 * v / total) if total else 0.0:.1f}%"
                 for k, v in items]
        line = f"[SR-prof] total={total:.1f}s"
        if parts:
            line += " | " + " | ".join(parts)
        if self.blocks:
            line += ("\n[SR-prof] blocks: "
                     + " ".join(f"b{i}={v:.2f}s" for i, v in enumerate(self.blocks)))
        return line


def _expand_anchor_kv_to_hr(anchor_kv, *, f, lq_hw, hr_hw):
    """Nearest-neighbour-expand anchor K/V harvested on the LQ grid back to HR token counts (scheme B).

    The native LQ-anchor harvest only runs `h_a * w_a` tokens, but the consumers
    (the window geometry in ``attention.py``, the triton kernel's params row)
    hard-assume the anchor grid == the HR grid. Scheme B therefore replicates
    every LQ token into an `s_h x s_w` HR block:

        LQ (i, j) ──> HR (s_h*i ... s_h*i+s_h-1, s_w*j ... s_w*j+s_w-1)

    Combined with the strided RoPE (LQ token i's RoPE position = HR's `s*i`),
    the key position seen by HR query `(s*i, s*j)` is exactly itself -- the
    anchor is a down-scaled GT, but addressed in HR coordinates.

    Replicating instead of dropping also has a direct benefit: the same key
    appears `s^2` times inside a window, equivalent to adding `log(s^2)` to
    every anchor logit, so the anchor's softmax mass stays at the magnitude it
    had at training time (HR token counts). The true-native scheme A would cut
    it by ~`s^2`, changing content and strength at once.

    Args:
        anchor_kv: per-layer `[(k, v), ...]`, each `[b, f*h_a*w_a, heads, head_dim]`.
        f: frames of this chunk (the token sequence is flattened row-major as
            `f, h_a, w_a`).
        lq_hw / hr_hw: **token** grids (= latent size // patch_size), not latent sizes.

    Returns:
        per-layer `[(k, v), ...]`, each `[b, f*h*w, heads, head_dim]`. When
        `s_h == s_w == 1` the input tensors are returned unchanged.
    """
    h_a, w_a = int(lq_hw[0]), int(lq_hw[1])
    h, w = int(hr_hw[0]), int(hr_hw[1])
    if h_a < 1 or w_a < 1 or h % h_a or w % w_a:
        raise ValueError(
            f"HR token grid ({h}, {w}) must be an integer multiple of the LQ "
            f"anchor token grid ({h_a}, {w_a}); this ratio is not divisible.")
    s_h, s_w = h // h_a, w // w_a
    expected = f * h_a * w_a
    out = []
    for k, v in anchor_kv:
        pair = []
        for t in (k, v):
            b, tok, nh, hd = t.shape
            if tok != expected:
                raise ValueError(
                    f"anchor carries {tok} tokens per sample but the LQ grid "
                    f"implies f*h_a*w_a = {expected} (f={f}, lq_hw={lq_hw}).")
            if s_h == 1 and s_w == 1:
                pair.append(t)
                continue
            g = t.view(b, f, h_a, w_a, nh, hd)
            g = g.repeat_interleave(s_h, dim=2).repeat_interleave(s_w, dim=3)
            pair.append(g.reshape(b, f * h * w, nh, hd))
        out.append((pair[0], pair[1]))
    return out


def triton_prewarm_variants(*, frames_per_block, use_lq_anchor, anchor_align,
                            prefill_frames=(), anchor_window_scope="window"):
    """``(f_cur, anchor_mode)`` pairs the streaming SR loop will actually launch.

    The prewarm hook needs the real list, not a guess: triton specializes the
    fused block-grid kernel on ``M_TILES == 1``, so a 1-frame anchor harvest is a
    different kernel from a 3-frame denoise chunk (a cold run compiles both,
    2263 ms + 1355 ms, inside the first DiT forwards). Missing a variant puts that
    compile back on the hot path; adding one that never runs is startup cost for
    nothing.

    The forwards, per block:

      * the LQ-anchor harvest — carries no anchor of its own, one frame at a time
        when frame-aligned, the whole chunk otherwise;
      * the denoise steps — anchor-carrying when the anchor is on;
      * the KV-refresh step — the same ``f_cur`` as the denoise steps but **no
        anchor** (it re-runs the denoised chunk at ``context_noise`` to write the
        cache), so with a frame-aligned anchor it is a THIRD specialization,
        distinct from both the anchored denoise and the 1-frame harvest.

    ``prefill_frames`` covers ``initial_latent``, which replays clean reference
    chunks with no anchor *before* the loop (1 frame then whole chunks when
    ``independent_first_frame``). Order is stable and duplicates are dropped.

    ``anchor_window_scope="global"`` drops the anchored forwards on purpose: the
    fused params row carries one spatial extent for both the anchor and the HR
    cuboid, so ``attn_impl="triton"`` raises on those calls rather than falling
    back — prewarming their kernel would compile something the run never launches.
    """
    anchor_mode = None
    if use_lq_anchor:
        anchor_mode = "frame" if anchor_align == "frame" else "chunk"
    fused = anchor_mode if anchor_window_scope == "window" else None
    if isinstance(frames_per_block, int):
        frames_per_block = [frames_per_block]
    out, seen = [], set()

    def add(v):
        if v not in seen:
            seen.add(v)
            out.append(v)

    for frames in frames_per_block:
        f = int(frames)
        if f <= 0:
            continue
        if anchor_mode is None or fused is not None:
            add((f, fused))                 # denoise steps
        if use_lq_anchor:                   # anchor harvest (never anchored)
            add((1 if anchor_align == "frame" else f, None))
        add((f, None))                      # KV refresh (never anchored)
    for frames in prefill_frames:           # clean reference chunks
        f = int(frames)
        if f > 0:
            add((f, None))
    return out


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            need_vae=True
    ):
        super().__init__()
        # Step 1: Initialize all models
        import os
        wan_model_dir = getattr(args, "wan_model_dir", "wan_models/Wan2.1-T2V-1.3B")
        self.generator = SRDiTWrapper(
            dit_config=args.dit_arch_config,
            checkpoint=getattr(args, "sr_dit_checkpoint", None),
            wan_model_dir=wan_model_dir,
        ) if generator is None else generator
        self.text_encoder = WanTextEncoder(
            wan_model_dir=wan_model_dir,
        ) if text_encoder is None else text_encoder
        if need_vae:
            if vae is not None:
                self.vae = vae
            elif getattr(args, "enable_nu_lightvae", False):
                # Distilled LightVAE-NU student (decode-only). Same Wan2.2
                # latent space; the decoder dominates end-to-end time at high
                # resolutions, so this is the largest lever on wall clock.
                from utils.sr_dit_wrapper import (NULightVAEWrapper22,
                                                 DEFAULT_NU_LIGHTVAE_MODULE)
                self.vae = NULightVAEWrapper22(
                    ckpt_path=getattr(args, "nu_lightvae_ckpt", None),
                    module_path=(getattr(args, "nu_lightvae_module_path", None)
                                 or DEFAULT_NU_LIGHTVAE_MODULE),
                    device=device,
                    dtype=torch.bfloat16,
                    nu_lightvae_type=getattr(args, "nu_lightvae_type", "scheme3"),
                )
            else:
                vae_version = getattr(args, "vae_version", "2.2")
                vae_pth = getattr(args, "vae_pth", None)
                if vae_version == "2.2":
                    if vae_pth is None:
                        vae_pth = os.path.join(wan_model_dir, "Wan2.2_VAE.pth")
                    self.vae = WanVAEWrapper22(vae_pth=vae_pth)
                else:
                    if vae_pth is None:
                        vae_pth = os.path.join(wan_model_dir, "Wan2.1_VAE.pth")
                    self.vae = WanVAEWrapper(vae_pth=vae_pth)

        # Optional trained latent upsampler; None -> _upsample_latent falls back
        # to bilinear. Enabled via latent_upsample_mode / latent_upsampler_arch_config
        # / latent_upsampler_ckpt (same fields as the diffusion SR pipelines).
        self.latent_upsampler = self._load_latent_upsampler(args)

        # Step 2: Initialize all causal hyperparameters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        if hasattr(args, "denoising_step_list_first_chunk") and args.denoising_step_list_first_chunk is not None:
            self.denoising_step_list_first_chunk = torch.tensor(
                args.denoising_step_list_first_chunk, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list_first_chunk = timesteps[1000 - self.denoising_step_list_first_chunk]
        else:
            self.denoising_step_list_first_chunk = None

        self.kv_caches = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.stream_kv_len = getattr(args, "stream_kv_len", None)
        # stream_kv_len <= 0 (e.g. -1) means "keep the entire KV cache" (no truncation).
        if self.stream_kv_len is not None and int(self.stream_kv_len) <= 0:
            self.stream_kv_len = None

        self.last_generation_time = None
        self.first_chunk_time = None

        # Stage split (pre / dit / decode) for every clip. `PROFILE_SR=1` adds the
        # per-block DiT detail, which costs a sync per block.
        self.stage_timer = StageTimer(
            sync=torch.cuda.synchronize,
            detail=os.environ.get("PROFILE_SR", "0") == "1")

        logging.info(f"KV inference with {self.num_frame_per_block} frames per block"
              f", kv_len={self.stream_kv_len}")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    # ------------------------------------------------------------------
    # Memory-lean VAE decode
    # ------------------------------------------------------------------
    def _decode_output(self, output, offload_for_decode=False, extra_offload_modules=None):
        """Free the (now-unused) streaming KV cache, then VAE-decode ``output``.

        The streaming KV cache is only needed for the autoregressive denoise
        loop; it is dropped here so its (potentially tens of GB at 2K/4K) memory
        is available for the high-resolution decoder. ``decode_to_pixel`` runs
        with ``offload_output=True`` so the decoded video accumulates on CPU.

        When ``offload_for_decode`` is True, idle models (this pipeline's
        generator and text encoder, plus any ``extra_offload_modules`` — e.g. the
        forcing stage's models in a cascade) are moved to CPU for the duration of
        the decode and restored afterwards. Modules are de-duplicated by identity
        (shared T5/VAE are handled once) and the VAE itself is never offloaded.
        """
        # KV cache is dead once the denoise loop finishes — free it before decode.
        self.kv_caches = None
        torch.cuda.empty_cache()

        offloaded = []  # (module, original_device)
        if offload_for_decode:
            seen = {id(self.vae)}
            candidates = [self.generator, self.text_encoder] + list(extra_offload_modules or [])
            for m in candidates:
                if m is None or id(m) in seen:
                    continue
                seen.add(id(m))
                orig_dev = next(m.parameters()).device
                if orig_dev.type == "cpu":
                    continue
                m.to("cpu")
                offloaded.append((m, orig_dev))
            if offloaded:
                torch.cuda.empty_cache()

        # Only the decoder call is billed to `decode`: the KV free, the offload
        # shuffle and the empty_cache above are memory management whose cost does
        # not change when the decoder does, so they stay visible in `other` rather
        # than inflating the number the VAE knobs are compared on.
        with self.stage_timer.stage("decode"):
            video = self.vae.decode_to_pixel(output, offload_output=True)

        for m, orig_dev in offloaded:
            m.to(orig_dev)
        if offloaded:
            torch.cuda.empty_cache()

        return video
    @staticmethod
    def _load_latent_upsampler(args) -> Optional[torch.nn.Module]:
        """Build the trained latent upsampler from config, or None for bilinear.

        Mirrors the diffusion SR pipelines: reads latent_upsample_mode,
        latent_upsampler_arch_config, latent_upsampler_precision, and
        latent_upsampler_ckpt off ``args``.
        """
        mode = getattr(args, "latent_upsample_mode", "bilinear")
        if mode == "bilinear":
            return None
        upsampler_cfg = getattr(args, "latent_upsampler_arch_config", None)
        if not upsampler_cfg:
            raise ValueError("latent_upsampler_arch_config required when latent_upsample_mode != bilinear")

        target = upsampler_cfg.get("target")
        params = upsampler_cfg.get("params", {})

        module_path, cls_name = target.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        upsampler = cls(**params)

        precision_key = getattr(args, "latent_upsampler_precision", "bf16")
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        upsampler = upsampler.to(dtype=dtype_map.get(precision_key, torch.bfloat16))

        ckpt = getattr(args, "latent_upsampler_ckpt", None)
        if ckpt:
            from pathlib import Path
            if not Path(ckpt).exists():
                raise FileNotFoundError(
                    f"Latent upsampler ckpt not found: {ckpt}. Refusing to fall back "
                    f"to init weights — that silently runs an UNTRAINED upsampler "
                    f"(with init_mode=nearest it degrades to plain nearest interp).")
            ckpt_state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model_sd = ckpt_state.get("model", ckpt_state)
            upsampler.load_state_dict(model_sd, strict=False)
            del ckpt_state
            logging.info(f"[Latent Upsampler] Loaded checkpoint: {ckpt}")

        upsampler.eval().requires_grad_(False)
        return upsampler

    def _upsample_latent(self, lr_latent: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        b, f, c, h, w = lr_latent.shape
        if h == target_h and w == target_w:
            return lr_latent
        # Trained upsampler (spatial x-scale baked into the model) if configured.
        if self.latent_upsampler is not None:
            model_dtype = next(self.latent_upsampler.parameters()).dtype
            x = lr_latent.permute(0, 2, 1, 3, 4).to(dtype=model_dtype)  # [B, C, T, H, W]
            x = self.latent_upsampler(x)
            return x.permute(0, 2, 1, 3, 4).to(dtype=lr_latent.dtype)
        lr_flat = lr_latent.reshape(b * f, c, h, w)
        up = F.interpolate(lr_flat, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return up.reshape(b, f, c, target_h, target_w)

    def _extract_anchor_kv_per_frame(self, lq_chunk, conditional_dict,
                                     harvest_t, base_offset, cond_y=None,
                                     per_frame=True, rope_hw_stride=None):
        """Per-frame (cf=1) or per-chunk LR anchor K/V harvest for the current chunk.

        Frame-aligned anchor (HR frame i ↔ LR frame i): with ``per_frame=True``
        (default) each LR frame is run through the DiT as an INDEPENDENT 1-frame
        sample (fresh cache), so frame i's K/V derive from LR frame i ONLY — no
        cross-frame leakage inside the chunk. This matches the self-forcing
        training harvest and the TF warm-start's model/sr_diffusion.py::_extract_anchor_kv.

        With ``per_frame=False`` the WHOLE chunk is run through one forward
        (chunk-internal frames attend each other during the harvest), matching
        the per-chunk training/inference semantics of Causal-Forcing-VSR.

        The (clean) LR is fed with a timestep embedding = ``harvest_t`` AND is actually
        corrupted with the matching flow-matching noise ``(1-σ)·lr + σ·eps``
        (σ = harvest_t / num_train_timesteps) before harvest, so input-noise-level and
        conditioning-timestep are self-consistent — IDENTICAL to training
        (self_forcing) and the TF warm-start. ``base_offset`` shifts the RoPE
        temporal origin so the anchor K/V align 1:1 with the HR query tokens of this
        chunk. Returns a per-layer list of ``(k, v)`` each shaped
        ``[B, F*frame_tokens, heads, dim]`` — identical shape/order either way.

        ``rope_hw_stride`` (native LQ-anchor): the harvested K is already RoPE'd,
        so a non-upsampled ``lq_chunk`` must get its RoPE at HR positions with a
        coarser step — ``(s_h, s_w)`` puts anchor row i at HR row ``i*s_h``. None
        (default) keeps the HR-grid harvest exactly as before.
        """
        b, f, c, h, w = lq_chunk.shape
        if per_frame:
            cf = 1
            nc = f
            lr_frames = lq_chunk.reshape(b, nc, cf, c, h, w).reshape(b * nc, cf, c, h, w)
        else:
            cf = f
            nc = 1
            lr_frames = lq_chunk

        # Corrupt the LR anchor with `harvest_t` steps of flow-matching noise
        # (σ = harvest_t / num_train_timesteps), matching training self_forcing and the
        # TF warm-start's _extract_anchor_kv. harvest_t=0 -> clean anchor (no-op).
        ctx = int(harvest_t)
        if ctx > 0:
            sigma = float(ctx) / float(self.scheduler.num_train_timesteps)
            lr_frames = (1.0 - sigma) * lr_frames + sigma * torch.randn_like(lr_frames)

        prompt_embeds = conditional_dict["prompt_embeds"]
        anchor_cond = {"prompt_embeds": prompt_embeds.repeat_interleave(nc, dim=0)}

        device = lq_chunk.device
        timestep = torch.ones(
            [b * nc, cf], device=device, dtype=torch.int64) * int(harvest_t)
        if per_frame:
            temporal_offset = [base_offset + i * cf for i in range(nc)] * b
        else:
            # Chunk-level offset; the RoPE grid positions frame j at base_offset+j.
            temporal_offset = [base_offset] * b

        cond_y_frames = None
        if cond_y is not None:
            tail = cond_y.shape[2:]
            cond_y_frames = cond_y.reshape(b, nc, cf, *tail).reshape(b * nc, cf, *tail)

        _, _, kv_caches = self.generator(
            noisy_image_or_video=lr_frames,
            conditional_dict=anchor_cond,
            timestep=timestep,
            kv_caches=None,
            is_stream=True,
            temporal_offset=temporal_offset,
            cond_y=cond_y_frames,
            kv_len=self.stream_kv_len,
            rope_hw_stride=rope_hw_stride,
        )

        anchor_kv = []
        for cache_k, cache_v in kv_caches:
            _, tok, nh, hd = cache_k.shape
            k = cache_k.reshape(b, nc, tok, nh, hd).reshape(b, nc * tok, nh, hd)
            v = cache_v.reshape(b, nc, tok, nh, hd).reshape(b, nc * tok, nh, hd)
            anchor_kv.append((k, v))
        return anchor_kv

    def _prewarm_window_triton(self, *, frames_per_block, height, width,
                               use_lq_anchor, anchor_align,
                               anchor_window_scope, prefill_frames=(),
                               device=None, anchor_latent_hw=None,
                               anchor_hw=None):
        """Compile the fused block-grid kernels before any real forward runs.

        Fires ONLY for ``window_attn_impl == "triton"``: the other three backends
        must keep a bit-identical startup path, or a 4-arm benchmark compares
        startups as well as backends. ``height``/``width`` are LATENT dims, so the
        token grid the op sees is ``latent // patch_size``.

        ``anchor_latent_hw`` (native LQ-anchor): the harvest forward runs on the
        LQ grid, which the HR pass above does not cover, so its kernel would
        compile mid-run — inside the very ``[SR-prof] dit`` number the
        native/baseline A/B is read off. A second pass warms JUST the harvest
        variant there; skipped when the two grids coincide.

        ``anchor_hw`` (native scheme "true" only) is the anchor's TOKEN grid, which
        the HR denoise forwards then attend as a second geometry: it changes
        ``HR_BASE`` and therefore the specialization, so the prewarm has to know
        it. The LQ harvest pass below carries no anchor and is unaffected.

        Returns the variants warmed ([] when this is not a triton run).
        """
        mod = next((m for m in self.generator.model.modules()
                    if getattr(m, "use_window_attn", False)
                    and getattr(m, "window_attn_impl", None) == "triton"
                    and not getattr(m, "window_chunk", None)), None)
        if mod is None:
            return []
        patch = tuple(getattr(self.generator.model, "patch_size", (1, 1, 1)))
        variants = triton_prewarm_variants(
            frames_per_block=frames_per_block, use_lq_anchor=use_lq_anchor,
            anchor_align=anchor_align, prefill_frames=prefill_frames,
            anchor_window_scope=anchor_window_scope)
        param = next(mod.parameters(), None)
        common = dict(
            # same fallback as the op (models.py: window_block_t or the chunk)
            bt=int(mod.window_block_t if mod.window_block_t is not None
                   else mod.stream_chunk_size),
            bh=int(mod.window_block_hw[0]), bw=int(mod.window_block_hw[1]),
            rh=int(mod.window_block_radius_hw[0]),
            rw=int(mod.window_block_radius_hw[1]),
            block_radius_t=mod.window_radius_t,
            n=int(mod.num_heads), d=int(mod.head_dim),
            device=device, dtype=(param.dtype if param is not None
                                  else torch.bfloat16))
        t0 = time.time()
        warmed = prewarm_blockgrid_triton(
            h=height // patch[1], w=width // patch[2], variants=variants,
            anchor_hw=anchor_hw, **common)
        total = len(variants)
        if (use_lq_anchor and anchor_latent_hw is not None
                and tuple(anchor_latent_hw) != (height, width)):
            # Only the harvest runs on the LQ grid; the denoise / KV-refresh /
            # prefill forwards stay on HR, so warming them here would compile
            # kernels this run never launches.
            if anchor_align == "frame":
                lq_variants = [(1, None)]        # cf=1, one forward per LR frame
            else:                                # one forward per chunk length
                fpb = ([frames_per_block] if isinstance(frames_per_block, int)
                       else list(frames_per_block))
                lq_variants = [(f, None) for f in
                               dict.fromkeys(int(x) for x in fpb if int(x) > 0)]
            warmed = list(warmed) + prewarm_blockgrid_triton(
                h=int(anchor_latent_hw[0]) // patch[1],
                w=int(anchor_latent_hw[1]) // patch[2],
                variants=lq_variants, **common)
            total += len(lq_variants)
        logging.info(f"[Prewarm] block-grid triton: {len(warmed)}/{total} "
              f"variants {warmed} in {time.time() - t0:.2f}s")
        return warmed

    def _prewarm_fp8_quantizer(self, *, frames_per_block, height, width,
                               prefill_frames=(), device=None):
        """Compile the fp8 activation quantiser before any real forward runs.

        Same argument as the block-grid prewarm: the quantiser is
        ``torch.compile``d per shape (``dynamic=False`` is 5.4x eager), and
        inductor's ~1.6 s-per-shape compile would otherwise land inside the first
        DiT forwards -- where it reads as fp8 being SLOWER than bf16 on a cold
        process. Silent no-op unless ``--fp8_linear`` actually converted
        something, so the pipeline may call it unconditionally.

        ``height``/``width`` are LATENT dims; the rows the Linears see are token
        counts, i.e. ``latent // patch_size``.

        Returns the shapes warmed ([] on a bf16 run).
        """
        from wan.modules.sr_dit.fp8_linear import (
            Fp8Linear, prewarm_quantizer, quantizer_shapes)
        model = self.generator.model
        if not any(isinstance(m, Fp8Linear) for m in model.modules()):
            return []
        patch = tuple(getattr(model, "patch_size", (1, 1, 1)))
        frame_tokens = (height // patch[1]) * (width // patch[2])
        # The prefill (`initial_latent`) forwards are REAL launches that happen
        # before the first denoise chunk, and with independent_first_frame they
        # run a 1-frame sample that `frames_per_block` does not contain -- warming
        # only the denoise sizes would move that compile into the prefill instead
        # of removing it (the same correction the triton prewarm needed).
        fs = {int(f) for f in frames_per_block} | {int(f) for f in prefill_frames}
        # Denoise runs f frames as one sample; the per-frame anchor harvest runs
        # f 1-frame samples batched -- both reach the Linears as f * frame_tokens
        # rows, so one set covers both.
        tok = {f * frame_tokens for f in fs}
        text_len = int(getattr(model, "text_len", 512))
        # cross_attn.k/v see the TEXT context, not the video tokens: length
        # text_len in a denoise forward, repeat_interleave'd to nc * text_len in
        # the per-frame harvest (_extract_anchor_kv above).
        text_tok = {text_len} | {f * text_len for f in fs}
        shapes = quantizer_shapes(tok, int(model.dim), int(model.ffn_dim), text_tok)
        param = next(model.parameters(), None)
        t0 = time.time()
        done = prewarm_quantizer(
            shapes, device=device,
            dtype=(param.dtype if param is not None else torch.bfloat16))
        logging.info(f"[Prewarm] fp8 quantiser: {len(done)}/{len(shapes)} shapes "
              f"{done} in {time.time() - t0:.2f}s")
        return done

    def inference_sr(
        self,
        lr_latent: torch.Tensor,
        noise: torch.Tensor,
        text_prompts: List[str],
        sigma_start: float = 0.9,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        return_video: bool = True,
        conditional_dict: Optional[dict] = None,
        cond_noise_sigma: float = 0.0,
        offload_for_decode: bool = False,
        extra_offload_modules: Optional[list] = None,
        use_lq_anchor: bool = False,
        lq_guidance_scale=(1.0,),
        lq_guidance_mode: str = "v",
        anchor_context_noise: Optional[int] = None,
        anchor_layers: Optional[list] = None,
        anchor_align: str = "frame",
        anchor_window_scope: str = "window",
        native_lq_anchor: bool = False,
        native_anchor_scheme: str = "repeat",
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        _mark = self.stage_timer.tick()

        # DMD few-step inference matches its self-forcing rollout by harvesting
        # the anchor at a dedicated noise level (not the KV-refresh context_noise)
        # and optionally restricting anchor injection to a subset of blocks. Both
        # default to the original behaviour: anchor_context_noise=None falls back
        # to self.args.context_noise; anchor_layers=None injects into all layers.
        anchor_harvest_t = (self.args.context_noise
                            if anchor_context_noise is None
                            else int(anchor_context_noise))

        lr_up = self._upsample_latent(lr_latent, height, width)
        sr_noise = (1.0 - sigma_start) * lr_up + sigma_start * noise

        # ── LQ-anchor: per-chunk low-res KV context (optional; default OFF) ──
        # When enabled, each chunk's denoising steps additionally attend to the K/V of
        # the upsampled-LR (LQ) version of the same chunk. The anchor is injected for the
        # attention computation only and never written into the rolling KV cache, so the
        # temporal streaming behaviour across chunks is identical to when it is disabled.
        if isinstance(lq_guidance_scale, (int, float)):
            lq_guidance_scale = [float(lq_guidance_scale)]
        else:
            lq_guidance_scale = [float(s) for s in lq_guidance_scale] or [1.0]
        if use_lq_anchor:
            logging.info(f"[SR] lq_anchor=on (mode={lq_guidance_mode}, "
                  f"scale={lq_guidance_scale}, harvest_t={anchor_harvest_t}, "
                  f"layers={anchor_layers}, align={anchor_align}, "
                  f"window_scope={anchor_window_scope})")
        if anchor_align not in ("frame", "chunk"):
            raise ValueError(f"anchor_align must be 'frame' or 'chunk', got {anchor_align!r}.")
        # Anchor SPATIAL scope under window attention: "window" = each window /
        # block attends only the anchor tokens in its own spatial extent (must
        # match training), "global" = legacy whole-frame/chunk anchor. Inert when
        # window attention is off.
        if anchor_window_scope not in ("window", "global"):
            raise ValueError(
                f"anchor_window_scope must be 'window' or 'global', got "
                f"{anchor_window_scope!r}.")

        # refiner_concat: feed the (optionally noise-augmented) upsampled LR as the
        # channel-concat condition cond_y (generator in_dim=96). Default "refiner"
        # mode leaves cond_y_full=None so behaviour is unchanged.
        use_concat = getattr(self.args, "conditioning_mode", "refiner") == "refiner_concat"
        if use_concat and cond_noise_sigma > 0:
            cond_noise = torch.randn_like(lr_up)
            cond_y_full = (1.0 - cond_noise_sigma) * lr_up + cond_noise_sigma * cond_noise
        elif use_concat:
            cond_y_full = lr_up
        else:
            cond_y_full = None

        # ── native LQ-anchor: harvest on the LQ grid ─────────────────────────────
        # The anchor's RoPE is baked into K at harvest time, so the small latent must
        # be RoPE'd at HR positions with a coarser step (rope_hw_stride). Two schemes
        # then differ only in what happens to the harvested K/V:
        #   "repeat" (B, default) — nearest-neighbour expand back to HR token counts,
        #       so every consumer keeps seeing one grid. Each key appears s^2 times,
        #       which is +log(s^2) on every anchor logit, i.e. the anchor keeps the
        #       softmax mass the deployed LoRA was distilled with.
        #   "true"   (A) — keep the LQ token count and hand the op the anchor's own
        #       grid (anchor_cfg['hw']). Cheaper and honest about the geometry, but it
        #       also divides the anchor's attention mass by ~s^2.
        # All geometry is in TOKENS (latent // patch_size) because that is what
        # grid_sizes / the window ops count.
        if native_anchor_scheme not in ("repeat", "true"):
            raise ValueError(
                f"native_anchor_scheme must be 'repeat' or 'true', got "
                f"{native_anchor_scheme!r}. A typo here would silently run the "
                f"other scheme and be reported under the wrong tag.")
        native_true = native_lq_anchor and native_anchor_scheme == "true"
        anchor_rope_stride = None
        anchor_lq_tok = anchor_hr_tok = None
        if use_lq_anchor and native_lq_anchor:
            if use_concat:
                raise ValueError(
                    "native_lq_anchor is not supported with "
                    "conditioning_mode='refiner_concat': cond_y lives on the HR "
                    "grid and has no defined meaning for a LQ-grid harvest.")
            patch = tuple(getattr(self.generator.model, "patch_size", (1, 1, 1)))
            h_lq, w_lq = int(lr_latent.shape[-2]), int(lr_latent.shape[-1])
            anchor_hr_tok = (height // patch[1], width // patch[2])
            anchor_lq_tok = (h_lq // patch[1], w_lq // patch[2])
            if (anchor_lq_tok[0] < 1 or anchor_lq_tok[1] < 1
                    or anchor_hr_tok[0] % anchor_lq_tok[0]
                    or anchor_hr_tok[1] % anchor_lq_tok[1]):
                raise ValueError(
                    f"native_lq_anchor needs an integer token-grid ratio: HR "
                    f"{anchor_hr_tok} vs LQ {anchor_lq_tok} (latent HR "
                    f"{(height, width)}, LQ {(h_lq, w_lq)}, patch {patch}).")
            anchor_rope_stride = (anchor_hr_tok[0] // anchor_lq_tok[0],
                                  anchor_hr_tok[1] // anchor_lq_tok[1])
            logging.info(
                f"[SR] lq_anchor native=on (scheme={native_anchor_scheme}): "
                f"harvest on LQ tokens {anchor_lq_tok} "
                + (f"-> kept native, anchor_cfg['hw']={anchor_lq_tok} "
                   f"(HR {anchor_hr_tok})" if native_true
                   else f"-> repeat to HR {anchor_hr_tok}")
                + f", rope_hw_stride={anchor_rope_stride}")

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        if conditional_dict is None:
            conditional_dict = self.text_encoder(text_prompts=text_prompts)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        self.kv_caches = None
        temporal_offset = 0

        # The per-block denoise chunk sizes, needed here (before the prefill
        # forwards below, which are real launches) by the triton prewarm.
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        self._prewarm_window_triton(
            frames_per_block=all_num_frames, height=height, width=width,
            use_lq_anchor=use_lq_anchor, anchor_align=anchor_align,
            anchor_window_scope=anchor_window_scope,
            prefill_frames=(() if initial_latent is None
                            else (((1,) if self.independent_first_frame else ())
                                  + (self.num_frame_per_block,))),
            device=noise.device,
            anchor_latent_hw=(tuple(lr_latent.shape[-2:])
                              if native_lq_anchor else None),
            # scheme="true" only: the HR denoise forwards then see an anchor on the
            # coarse grid, which moves HR_BASE and hence the specialization.
            anchor_hw=(anchor_lq_tok if native_true else None))
        self._prewarm_fp8_quantizer(
            frames_per_block=all_num_frames, height=height, width=width,
            prefill_frames=(() if initial_latent is None
                            else (((1,) if self.independent_first_frame else ())
                                  + (self.num_frame_per_block,))),
            device=noise.device)

        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                _, _, self.kv_caches = self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict, timestep=timestep * 0,
                    kv_caches=self.kv_caches, is_stream=True,
                    temporal_offset=temporal_offset,
                    kv_len=self.stream_kv_len)
                current_start_frame += 1
                temporal_offset += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[
                    :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                _, _, self.kv_caches = self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict, timestep=timestep * 0,
                    kv_caches=self.kv_caches, is_stream=True,
                    temporal_offset=temporal_offset,
                    kv_len=self.stream_kv_len)
                current_start_frame += self.num_frame_per_block
                temporal_offset += self.num_frame_per_block

        max_timestep = sigma_start * 1000
        sr_step_list = self.denoising_step_list[self.denoising_step_list <= max_timestep + 1e-3]
        if len(sr_step_list) == 0:
            sr_step_list = self.denoising_step_list[-1:]
        sr_step_list_first = None
        if self.denoising_step_list_first_chunk is not None:
            sr_step_list_first = self.denoising_step_list_first_chunk[
                self.denoising_step_list_first_chunk <= max_timestep + 1e-3]
            if len(sr_step_list_first) == 0:
                sr_step_list_first = self.denoising_step_list_first_chunk[-1:]

        # Log the effective per-block SR denoising schedule (sigma-truncated).
        logging.info(f"[SR] blocks={len(all_num_frames)} (frames/block={self.num_frame_per_block}), "
              f"latent {height}x{width}, sigma_start={sigma_start} -> max_t={max_timestep:.0f}, "
              f"full_schedule={self.denoising_step_list.tolist()} -> per-block denoise steps={sr_step_list.tolist()} "
              f"(+1 KV-refresh at t={self.args.context_noise}), "
              f"kv_len={self.stream_kv_len} chunks (None=unbounded, "
              f"{'sliding window' if self.stream_kv_len is not None else 'full history'})")
        frame_tokens = height * width  # latent tokens per frame, used for the per-block KV log

        # `pre` = text encode, latent upsample, JIT prewarm and the initial_latent
        # prefill forwards. Kept separate from `dit` because the prewarm is a
        # one-off compile bill and the prefill is not part of the denoise schedule;
        # folding either into `dit` would make the first clip of a run look slow for
        # a reason that has nothing to do with the attention backend.
        _mark = self.stage_timer.lap("pre", _mark)

        for block_index, current_num_frames in enumerate(tqdm.tqdm(all_num_frames)):
            _blk = self.stage_timer.tick() if self.stage_timer.detail else None
            noisy_input = sr_noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            chunk_cond_y = (
                cond_y_full[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                if cond_y_full is not None else None)

            current_denoising_list = (
                sr_step_list_first
                if block_index == 0 and sr_step_list_first is not None
                else sr_step_list
            )

            # Capture the LQ chunk's per-layer K/V for the current chunk. Shape
            # [B, F*tok, ...] is identical for both harvest modes; `anchor_align`
            # selects per-frame (cf=1, no cross-frame leakage) or per-chunk (one
            # forward over the whole chunk, matching Causal-Forcing-VSR).
            anchor_kv = None
            if use_lq_anchor:
                if native_lq_anchor:
                    # Same time-axis slice: _upsample_latent only touches h/w.
                    lq_chunk = lr_latent[
                        :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                else:
                    lq_chunk = lr_up[
                        :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                anchor_kv = self._extract_anchor_kv_per_frame(
                    lq_chunk, conditional_dict,
                    harvest_t=anchor_harvest_t,
                    base_offset=temporal_offset,
                    cond_y=chunk_cond_y,
                    per_frame=(anchor_align == "frame"),
                    rope_hw_stride=anchor_rope_stride)
                if native_lq_anchor and not native_true:
                    # scheme "repeat": undo the token-count difference here so the
                    # op keeps seeing one grid. scheme "true" carries the coarse
                    # grid into attention instead (anchor_cfg['hw'] below).
                    anchor_kv = _expand_anchor_kv_to_hr(
                        anchor_kv, f=lq_chunk.shape[1],
                        lq_hw=anchor_lq_tok, hr_hw=anchor_hr_tok)

            for index, current_timestep in enumerate(current_denoising_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device, dtype=torch.int64) * current_timestep

                anchor_cfg = None
                if use_lq_anchor:
                    scale = (lq_guidance_scale[index]
                             if index < len(lq_guidance_scale) else lq_guidance_scale[-1])
                    anchor_cfg = {"scale": float(scale), "mode": lq_guidance_mode,
                                  "align": anchor_align,
                                  "window_scope": anchor_window_scope}
                    if anchor_layers is not None:
                        anchor_cfg["layers"] = anchor_layers
                    if native_true:
                        # The anchor's OWN token grid: the op needs it to build the
                        # anchor's window columns (Task 8). Absent, it would assert
                        # on the f_cur*h*w token count.
                        anchor_cfg["hw"] = anchor_lq_tok

                if index < len(current_denoising_list) - 1:
                    _, denoised_pred, _ = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict, timestep=timestep,
                        kv_caches=self.kv_caches, is_stream=True,
                        temporal_offset=temporal_offset,
                        cond_y=chunk_cond_y,
                        kv_len=self.stream_kv_len,
                        anchor_kv=anchor_kv, anchor_cfg=anchor_cfg)
                    next_timestep = current_denoising_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred, _ = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict, timestep=timestep,
                        kv_caches=self.kv_caches, is_stream=True,
                        temporal_offset=temporal_offset,
                        cond_y=chunk_cond_y,
                        kv_len=self.stream_kv_len,
                        anchor_kv=anchor_kv, anchor_cfg=anchor_cfg)

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            _, _, self.kv_caches = self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict, timestep=context_timestep,
                kv_caches=self.kv_caches, is_stream=True,
                temporal_offset=temporal_offset,
                cond_y=chunk_cond_y,
                kv_len=self.stream_kv_len)

            # [KV] Confirm the streaming cache is actually truncated to kv_len chunks.
            # cache_k of layer 0 has shape [b, tokens, heads, dim]; tokens = frames * (h*w).
            # With kv_len=N the retained history should plateau at N chunks (= N*frames_per_block
            # latent frames) instead of growing every block.
            kv_tokens = self.kv_caches[0][0].shape[1]
            kv_frames = kv_tokens // frame_tokens
            kv_chunks = kv_frames / self.num_frame_per_block

            # logging.info(f"[KV] block {block_index:>2}/{len(all_num_frames) - 1}: "
            #       f"retained cache = {kv_chunks:g} chunks "
            #       f"({kv_frames} latent frames, {kv_tokens} tokens); "
            #       f"next chunk attends to these + itself "
            #       f"[cap kv_len={self.stream_kv_len}]")

            if _blk is not None:
                self.stage_timer.blocks.append(self.stage_timer.tick() - _blk)

            current_start_frame += current_num_frames
            temporal_offset += current_num_frames

        # `dit` = the denoise loop only, so the window backend and fp8 have one
        # number they alone move.
        _mark = self.stage_timer.lap("dit", _mark)

        if return_video:
            video = self._decode_output(
                output,
                offload_for_decode=offload_for_decode,
                extra_offload_modules=extra_offload_modules,
            )
            video = (video * 0.5 + 0.5).clamp(0, 1)
            if return_latents:
                return video, output
            return video
        return output

    def inference_sr_df(
        self,
        lr_latent: torch.Tensor,
        noise: torch.Tensor,
        text_prompts: List[str],
        sigma_start: float = 0.9,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        return_video: bool = True,
        conditional_dict: Optional[dict] = None,
        unconditional_dict: Optional[dict] = None,
        cond_noise_sigma: float = 0.0,
        lq_cfg: bool = False,
        use_flex_attention: bool = False,
    ) -> torch.Tensor:
        """Denoising-forcing causal few-step SR inference.

        Timestep outer loop, chunk inner loop. KV cache is rebuilt from
        scratch at each timestep so every chunk sees previous chunks'
        noisy representations at the *same* noise level.

        refiner_concat: when ``conditioning_mode == "refiner_concat"`` the
        (optionally noise-augmented) upsampled LR latent is fed as the
        channel-concat condition ``cond_y`` (generator in_dim=96). This matches
        the DMD ``sr_dmd_refiner_concat`` generator, which was trained with
        ``cond_noise_sigma`` augmentation on ``cond_y`` (the noisy x_t branch
        always keeps the clean ``lr_up`` endpoint).

        CFG (guidance_scale > 1) is optional and combines the x0 predictions of
        the conditional / unconditional passes (equivalent to combining flow,
        since x0 and flow are affine in x_t). ``lq_cfg`` guides on the presence
        of ``cond_y`` (zeroed for the uncond pass) instead of the text prompt.

        ``use_flex_attention`` runs a single full-sequence forward per denoising
        step (blockwise causal mask, no KV cache) instead of the per-chunk
        streaming loop.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        _mark = self.stage_timer.tick()

        lr_up = self._upsample_latent(lr_latent, height, width)

        use_concat = getattr(self.args, "conditioning_mode", "refiner") == "refiner_concat"

        # cond_y = (optionally noise-augmented) upsampled LR; the noisy x_t
        # branch (``latents``) always uses the clean lr_up endpoint.
        if use_concat and cond_noise_sigma > 0:
            cond_noise = torch.randn_like(lr_up)
            cond_y_full = (1.0 - cond_noise_sigma) * lr_up + cond_noise_sigma * cond_noise
        else:
            cond_y_full = lr_up
        latents = (1.0 - sigma_start) * lr_up + sigma_start * noise

        assert num_frames % self.num_frame_per_block == 0
        num_chunks = num_frames // self.num_frame_per_block
        chunk_size = self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        cond_y_all = cond_y_full if use_concat else None

        if conditional_dict is None:
            conditional_dict = self.text_encoder(text_prompts=text_prompts)

        guidance_scale = getattr(self.args, "guidance_scale", 1.0)
        do_cfg = guidance_scale is not None and guidance_scale > 1.0
        if do_cfg and unconditional_dict is None and not lq_cfg:
            unconditional_dict = self.text_encoder(
                text_prompts=[self.args.negative_prompt] * len(text_prompts))

        if use_flex_attention and initial_latent is not None:
            raise NotImplementedError("use_flex_attention does not support initial_latent")

        max_timestep = sigma_start * 1000
        sr_step_list = self.denoising_step_list[self.denoising_step_list <= max_timestep + 1e-3]
        if len(sr_step_list) == 0:
            sr_step_list = self.denoising_step_list[-1:]

        logging.info(f"[SR-df] chunks={num_chunks} (frames/block={chunk_size}), "
              f"latent {height}x{width}, sigma_start={sigma_start} -> max_t={max_timestep:.0f}, "
              f"steps={sr_step_list.tolist()}, concat={use_concat}, "
              f"cond_noise_sigma={cond_noise_sigma}, "
              f"cfg={guidance_scale if do_cfg else 'off'}"
              f"{' (lq_cfg)' if (do_cfg and lq_cfg) else ''}, "
              f"flex={use_flex_attention}, kv_len={self.stream_kv_len}")

        def _cache_initial(kv):
            """Prime a KV cache with the clean initial_latent frames at t=0."""
            cache_offset, cache_start = 0, 0
            t_zero = torch.zeros([batch_size, 1], device=noise.device, dtype=torch.int64)
            if self.independent_first_frame:
                assert (num_input_frames - 1) % chunk_size == 0
                n_blocks = (num_input_frames - 1) // chunk_size
                _, _, kv = self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict, timestep=t_zero,
                    kv_caches=kv, is_stream=True, temporal_offset=cache_offset,
                    kv_len=self.stream_kv_len)
                cache_start += 1
                cache_offset += 1
            else:
                assert num_input_frames % chunk_size == 0
                n_blocks = num_input_frames // chunk_size
            for _ in range(n_blocks):
                ref = initial_latent[:, cache_start:cache_start + chunk_size]
                _, _, kv = self.generator(
                    noisy_image_or_video=ref,
                    conditional_dict=conditional_dict, timestep=t_zero,
                    kv_caches=kv, is_stream=True, temporal_offset=cache_offset,
                    kv_len=self.stream_kv_len)
                cache_start += chunk_size
                cache_offset += chunk_size
            return kv

        # `pre` here is setup only — unlike the sf path, df rebuilds the KV cache
        # from scratch inside the loop (`_cache_initial`), so the prefill forwards
        # are genuinely part of `dit`.
        _mark = self.stage_timer.lap("pre", _mark)

        for step_idx, current_timestep in enumerate(tqdm.tqdm(sr_step_list)):
            # The outer loop is timesteps here, not chunks, so `blocks:` in the
            # report reads per denoising step on this path.
            _blk = self.stage_timer.tick() if self.stage_timer.detail else None
            if use_flex_attention:
                # Single full-sequence forward at the current noise level.
                timestep = torch.ones(
                    [batch_size, num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                # Full-sequence (no KV cache) -> wrapper returns (flow_pred, pred_x0).
                _, denoised_cond = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    cond_y=cond_y_all,
                )
                if do_cfg:
                    uncond_text = conditional_dict if lq_cfg else unconditional_dict
                    uncond_cond_y = torch.zeros_like(cond_y_all) if cond_y_all is not None else None
                    _, denoised_uncond = self.generator(
                        noisy_image_or_video=latents,
                        conditional_dict=uncond_text,
                        timestep=timestep,
                        cond_y=uncond_cond_y,
                    )
                    all_denoised = denoised_uncond + guidance_scale * (denoised_cond - denoised_uncond)
                else:
                    all_denoised = denoised_cond

                if step_idx < len(sr_step_list) - 1:
                    next_timestep = sr_step_list[step_idx + 1]
                    latents = self.scheduler.add_noise(
                        all_denoised.flatten(0, 1),
                        torch.randn_like(all_denoised.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, all_denoised.shape[:2])
                else:
                    latents = all_denoised
                if _blk is not None:
                    self.stage_timer.blocks.append(self.stage_timer.tick() - _blk)
                continue

            kv_caches = _cache_initial(None) if initial_latent is not None else None
            kv_neg = (_cache_initial(None) if (do_cfg and initial_latent is not None) else None)

            # Forward pass for each chunk at current_timestep
            denoised_chunks = []
            temporal_offset = num_input_frames
            for ci in range(num_chunks):
                cs = ci * chunk_size
                ce = cs + chunk_size
                chunk_noisy = latents[:, cs:ce]
                chunk_cond_y = cond_y_full[:, cs:ce] if use_concat else None
                timestep = torch.ones(
                    [batch_size, chunk_size],
                    device=noise.device, dtype=torch.int64) * current_timestep

                _, denoised_cond, kv_caches = self.generator(
                    noisy_image_or_video=chunk_noisy,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_caches=kv_caches,
                    is_stream=True,
                    temporal_offset=temporal_offset,
                    cond_y=chunk_cond_y,
                    kv_len=self.stream_kv_len,
                )
                if do_cfg:
                    uncond_text = conditional_dict if lq_cfg else unconditional_dict
                    uncond_cond_y = torch.zeros_like(chunk_cond_y) if chunk_cond_y is not None else None
                    _, denoised_uncond, kv_neg = self.generator(
                        noisy_image_or_video=chunk_noisy,
                        conditional_dict=uncond_text,
                        timestep=timestep,
                        kv_caches=kv_neg,
                        is_stream=True,
                        temporal_offset=temporal_offset,
                        cond_y=uncond_cond_y,
                        kv_len=self.stream_kv_len,
                    )
                    denoised_pred = denoised_uncond + guidance_scale * (denoised_cond - denoised_uncond)
                else:
                    denoised_pred = denoised_cond
                denoised_chunks.append(denoised_pred)
                temporal_offset += chunk_size

            all_denoised = torch.cat(denoised_chunks, dim=1)

            if step_idx < len(sr_step_list) - 1:
                next_timestep = sr_step_list[step_idx + 1]
                latents = self.scheduler.add_noise(
                    all_denoised.flatten(0, 1),
                    torch.randn_like(all_denoised.flatten(0, 1)),
                    next_timestep * torch.ones(
                        [batch_size * num_frames], device=noise.device, dtype=torch.long)
                ).unflatten(0, all_denoised.shape[:2])
            else:
                latents = all_denoised

            if _blk is not None:
                self.stage_timer.blocks.append(self.stage_timer.tick() - _blk)

        _mark = self.stage_timer.lap("dit", _mark)

        output = torch.zeros(
            [batch_size, num_frames + num_input_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)
        if initial_latent is not None:
            output[:, :num_input_frames] = initial_latent
        output[:, num_input_frames:] = latents

        if return_video:
            video = self._decode_output(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)
            if return_latents:
                return video, output
            return video
        return output
