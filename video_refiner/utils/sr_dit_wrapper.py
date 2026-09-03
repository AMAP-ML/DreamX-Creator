"""SR-DiT model wrappers.

SRDiTWrapper: wraps the SR-DiT Transformer3DModel with flow_pred -> x0 conversion,
              checkpoint loading (including zero-init patch_embedding expansion),
              and gradient checkpointing.
              Interface-compatible with the training-side diffusion wrapper.

WanVAEWrapper22: Wan2.2 VAE (z_dim=48) for online encode/decode.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.sr_dit.models import Transformer3DModel


class WanVAEWrapper22(nn.Module):
    """Wan2.2 VAE wrapper (z_dim=48, c_dim=160) for online encode/decode."""

    # False here / True on the student decoders below. Callers that need to
    # encode (SR must encode the LR video) branch on this instead of on a
    # per-decoder CLI flag, so a new decode-only decoder is handled for free.
    decode_only = False

    MEAN = [
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
        -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
        -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
        -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
        0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
    ]
    STD = [
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    ]

    def __init__(self, vae_pth: str):
        super().__init__()
        from wan.modules.vae2_2 import _video_vae
        self.mean = torch.tensor(self.MEAN, dtype=torch.float32)
        self.std = torch.tensor(self.STD, dtype=torch.float32)
        self.z_dim = 48
        self.model = _video_vae(
            pretrained_path=vae_pth, z_dim=48, dim=160,
            dim_mult=[1, 2, 4, 4], temperal_downsample=[False, True, True],
        ).eval().requires_grad_(False)

    @property
    def scale(self):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return [self.mean.to(device=device, dtype=dtype),
                1.0 / self.std.to(device=device, dtype=dtype)]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """pixel: [B, C, T, H, W] in [-1, 1] → latent [B, T, C_lat, H/8, W/8]"""
        scale = self.scale
        output = [self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False,
                        offload_output: bool = False) -> torch.Tensor:
        """latent: [B, T, C, H, W] → pixel [B, T, C_rgb, H*8, W*8] in [-1, 1]

        `use_cache` is accepted for interface parity with the 2.1 VAE wrapper
        (this 2.2 decoder does not use a temporal cache) and is ignored.

        `offload_output`: when True, the decoder accumulates each temporal chunk
        on CPU (instead of the GPU compute device) and the float/clamp is done on
        CPU, so the full high-resolution video never sits on the GPU. The result
        is returned as a CPU tensor. Default False keeps the original GPU path
        (used by training). See ``WanVAE_.decode(out_device=...)``.
        """
        zs = latent.permute(0, 2, 1, 3, 4)
        model_dtype = next(self.model.parameters()).dtype
        scale = self.scale
        out_device = "cpu" if offload_output else None
        output = []
        for u in zs:
            output.append(
                self.model.decode(u.to(model_dtype).unsqueeze(0), scale, out_device=out_device)
                .float().clamp_(-1, 1).squeeze(0)
            )
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------------------
# LightVAE-NU (distilled non-uniform-channel Wan2.2 decoder student)
# ---------------------------------------------------------------------------

# This repo vendors the `lightvae_nu/` package. The package binds its Wan VAE
# building blocks relative to its own location
# (`vae_blocks._VAE22_PATH` = `<module_path>/wan/modules/vae2_2.py`), so this
# default (derived from __file__) always resolves against THIS repo's
# `wan/modules/vae2_2.py`.
DEFAULT_NU_LIGHTVAE_MODULE = str(Path(__file__).resolve().parent.parent)

# scheme -> checkpoint. Only the released scheme3 continuation is wired up as a
# default; other schemes work with an explicit --nu_lightvae_ckpt.
DEFAULT_NU_LIGHTVAE_CKPT = {
    "scheme3": str(Path(__file__).resolve().parent.parent.parent
                   / "checkpoints" / "refiner" / "lightvae_nu_scheme3.pt"),
}


def load_lightvae_nu(module_path: str = DEFAULT_NU_LIGHTVAE_MODULE):
    """Import the ``lightvae_nu`` package from ``module_path`` and return it with
    ``.student`` / ``.channels`` attached.

    Loading by path rather than plain ``import lightvae_nu`` so that a same-named
    package elsewhere on ``sys.path`` cannot shadow the vendored copy: we prepend
    ``module_path``, drop any already-imported copy from ``sys.modules``, import,
    and then *verify* which copy answered -- a shadow raises rather than quietly
    building the decoder from another tree's source.
    """
    student_py = os.path.join(module_path, "lightvae_nu", "student.py")
    if not os.path.exists(student_py):
        raise FileNotFoundError(f"lightvae_nu package not found: {student_py}")

    if module_path in sys.path:
        sys.path.remove(module_path)
    sys.path.insert(0, module_path)

    want = Path(module_path).resolve()
    # A previously-imported (possibly vendored) copy wins over sys.path, so drop it.
    for name in [n for n in sys.modules if n == "lightvae_nu"
                 or n.startswith("lightvae_nu.")]:
        mod = sys.modules[name]
        f = getattr(mod, "__file__", None)
        if f is None or want not in Path(f).resolve().parents:
            del sys.modules[name]

    pkg = importlib.import_module("lightvae_nu")
    got = Path(pkg.__file__).resolve()
    if want not in got.parents:
        raise ImportError(
            f"lightvae_nu resolved to {got}, not under module_path={module_path}. "
            f"A same-named copy shadowed the real package; remove it from "
            f"sys.path or pass a different --nu_lightvae_module_path.")
    pkg.student = importlib.import_module("lightvae_nu.student")
    pkg.channels = importlib.import_module("lightvae_nu.channels")
    return pkg


def resolve_nu_lightvae_dims(ckpt_path: str,
                             module_path: str = DEFAULT_NU_LIGHTVAE_MODULE,
                             override: Optional[List[int]] = None):
    """Recover the student's 5-tuple of stage channels for ``ckpt_path``.

    The architecture is *not* derivable from a class default, and
    ``config["schedule"]`` is not authoritative: scheme2b carries
    ``schedule="scheme2"`` next to its own ``dims=(256,256,256,192,96)``, so
    resolving by schedule would build scheme2's graph for it. Resolution order is
    therefore explicit ``override`` -> ``config["dims"]`` -> ``config["schedule"]``
    via ``channels.NAMED_SCHEDULES``. Whatever comes out is cross-checked against
    the weights themselves (``decoder.conv1`` out-channels and the head's
    RMS_norm gamma), so a wrong tuple raises here instead of half-loading a wrong
    graph.
    """
    pkg = load_lightvae_nu(module_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
    weights = ckpt.get("ema")
    if weights is None:
        raise ValueError(f"{ckpt_path} has no 'ema' key (keys: {list(ckpt)[:8]})")

    dims = override
    if dims is None:
        cfg = ckpt.get("config") or {}
        dims = cfg.get("dims")
        if dims is None:
            schedule = cfg.get("schedule")
            # No `{}` fallback: an absent table means an unexpected copy of the
            # package answered, and reporting "unknown schedule" would hide that.
            table = pkg.channels.NAMED_SCHEDULES
            if schedule not in table:
                raise ValueError(
                    f"{ckpt_path}: config has neither 'dims' nor a known "
                    f"'schedule' (got {schedule!r}; known: {sorted(table)})")
            dims = table[schedule]
    dims = tuple(int(d) for d in dims)

    first = int(weights["decoder.conv1.weight"].shape[0])
    last = int(weights["decoder.head.0.gamma"].shape[0])
    if (dims[0], dims[-1]) != (first, last):
        raise ValueError(
            f"{ckpt_path}: resolved dims {dims} contradict the weights "
            f"(conv1 out={first}, head width={last}). Pass the right --dims or "
            f"fix the checkpoint's config.")
    return dims


class NULightVAEWrapper22(nn.Module):
    """Drop-in replacement for :class:`WanVAEWrapper22` that decodes with a
    LightVAE-NU student (see :mod:`lightvae_nu`) instead of the full Wan2.2
    decoder.

    Same latent space as the full VAE (``z_dim=48``, spatial /16, temporal /4,
    same normalisation stats), so it plugs in decoder-only; encode is NOT
    supported (``inference_sr.py`` keeps a full Wan VAE for the LR encode, keyed
    on ``decode_only``). Includes per-frame CPU accumulation, which the
    package's own ``decode_causal`` lacks (it holds every decoded frame on GPU).

    ``nu_lightvae_type`` selects the distilled width; ``dims`` comes out of the
    checkpoint (see :func:`resolve_nu_lightvae_dims`), never out of a default.
    """

    decode_only = True

    # Same latent normalisation stats as the full Wan2.2 VAE. `lightvae_nu`
    # carries its own copy of these; tests pin the two together.
    MEAN = WanVAEWrapper22.MEAN
    STD = WanVAEWrapper22.STD

    def __init__(self, ckpt_path: Optional[str] = None,
                 module_path: str = DEFAULT_NU_LIGHTVAE_MODULE,
                 device="cpu", dtype=torch.bfloat16,
                 nu_lightvae_type: str = "scheme3",
                 dims: Optional[List[int]] = None):
        super().__init__()
        ckpt_path = ckpt_path or DEFAULT_NU_LIGHTVAE_CKPT.get(nu_lightvae_type)
        if ckpt_path is None:
            raise ValueError(
                f"No LightVAE-NU ckpt for type={nu_lightvae_type!r} "
                f"(known: {sorted(DEFAULT_NU_LIGHTVAE_CKPT)})")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"LightVAE-NU ckpt not found: {ckpt_path}")

        pkg = load_lightvae_nu(module_path)
        self._unpatchify = pkg.student.unpatchify
        self.dims = resolve_nu_lightvae_dims(ckpt_path, module_path=module_path,
                                             override=dims)

        self.mean = torch.tensor(self.MEAN, dtype=torch.float32)
        self.std = torch.tensor(self.STD, dtype=torch.float32)
        self.z_dim = 48

        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
        # `ema`, NOT `student`: both keys are present and the EMA weights are
        # the released ones.
        weights = ckpt["ema"]
        model = pkg.student.StudentVAE(dims=self.dims, z_dim=self.z_dim)
        model.load_state_dict(weights, strict=True)
        self.model = (model.eval().requires_grad_(False)
                      .to(device=device, dtype=dtype))
        self.model.clear_cache()
        step = ckpt.get("step")
        print(f"✅ LightVAE-NU ({nu_lightvae_type}, dims={self.dims}, "
              f"step={step}) loaded from: {ckpt_path}")

    @property
    def scale(self):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return [self.mean.to(device=device, dtype=dtype),
                1.0 / self.std.to(device=device, dtype=dtype)]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "NULightVAEWrapper22 is decode-only (distilled student decoder); "
            "use the full WanVAEWrapper22 if encoding is required.")

    def _decode_one(self, z: torch.Tensor, scale, out_device) -> torch.Tensor:
        """Decode one latent ``[1, C, T, H, W]`` -> pixel ``[1, 3, T', H*16, W*16]``.

        Same per-frame causal loop as ``StudentVAE.decode_causal`` (bit-identical
        to it), plus ``out_device`` CPU accumulation so the full-resolution
        video never sits on the GPU.
        """
        m = self.model
        m.clear_cache()
        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        x = m.conv2(z)
        out = None
        for i in range(x.shape[2]):
            m._conv_idx = [0]
            out_ = m.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=m._feat_map,
                feat_idx=m._conv_idx,
                first_chunk=(i == 0),
            )
            if out_device is not None:
                out_ = out_.to(out_device)
            out = out_ if out is None else torch.cat([out, out_], dim=2)
        out = self._unpatchify(out, patch_size=2)
        m.clear_cache()
        return out

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False,
                        offload_output: bool = False) -> torch.Tensor:
        """latent: [B, T, C, H, W] -> pixel [B, T, C_rgb, H*16, W*16] in [-1, 1].

        Signature/layout identical to :meth:`WanVAEWrapper22.decode_to_pixel`.
        ``use_cache`` is accepted for interface parity and ignored.
        """
        zs = latent.permute(0, 2, 1, 3, 4)
        model_dtype = next(self.model.parameters()).dtype
        scale = self.scale
        out_device = "cpu" if offload_output else None
        output = []
        for u in zs:
            output.append(
                self._decode_one(u.to(model_dtype).unsqueeze(0), scale, out_device)
                .float().clamp_(-1, 1).squeeze(0)
            )
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------------------
# Checkpoint loading utilities
# ---------------------------------------------------------------------------

def _load_dit_state_dict(checkpoint_path: str) -> dict:
    """Load DiT state_dict from a file or Wan model directory (safetensors shards)."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"DiT checkpoint not found: {checkpoint_path}")

    if path.is_dir():
        sf_files = sorted(path.glob("diffusion_pytorch_model*.safetensors"))
        if sf_files:
            from safetensors.torch import load_file
            sd = {}
            for sf in sf_files:
                sd.update(load_file(str(sf), device="cpu"))
            print(f"[SRDiT] Loaded {len(sf_files)} safetensors shards from {checkpoint_path}")
            return sd

        for pattern in ("model.pt", "*.pt", "*.pth"):
            matches = sorted(path.glob(pattern))
            if matches:
                path = matches[0]
                break
        else:
            raise FileNotFoundError(
                f"No DiT weights found in {checkpoint_path} "
                "(expected diffusion_pytorch_model*.safetensors or *.pt)"
            )

    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(str(path), device="cpu")
        print(f"[SRDiT] Loaded safetensors checkpoint: {path}")
        return sd

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    weight_keys = ["model", "generator", "generator_ema", "state_dict", "model_state_dict"]
    for key in weight_keys:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            print(f"[SRDiT] Loaded checkpoint key '{key}' from {path}")
            return ckpt[key]

    if isinstance(ckpt, dict):
        sample_keys = list(ckpt.keys())[:5]
        if any("weight" in k or "bias" in k or "embed" in k for k in sample_keys):
            print(f"[SRDiT] Loaded flat state_dict from {path}")
            return ckpt

    raise RuntimeError(f"Could not find state_dict in checkpoint: {path}")


def _normalize_dit_state_dict(sd: dict) -> dict:
    # Strip FSDP wrapper prefixes
    fsdp_prefixes = ["_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."]
    def _strip_fsdp(k):
        for prefix in fsdp_prefixes:
            k = k.replace(prefix, "")
        return k
    sd = {_strip_fsdp(k): v for k, v in sd.items()}

    if sd and all(k.startswith("model.") for k in sd.keys()):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    elif sd and any(k.startswith("model.") for k in sd.keys()):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    return sd


def _apply_patch_embedding_expand_zero_init(
    patch_embed: nn.Conv3d,
    ckpt_weight: torch.Tensor,
    base_in_dim: Optional[int] = None,
) -> None:
    """Copy pretrained input channels and zero-init expanded concat channels."""
    target_w = patch_embed.weight
    ckpt_in = ckpt_weight.shape[1]
    target_in = target_w.shape[1]

    if target_in == ckpt_in:
        target_w.data.copy_(ckpt_weight.to(device=target_w.device, dtype=target_w.dtype))
        return

    if base_in_dim is None:
        base_in_dim = ckpt_in

    if base_in_dim != ckpt_in:
        raise ValueError(
            f"base_in_dim={base_in_dim} does not match checkpoint patch_embedding "
            f"in_channels={ckpt_in}"
        )
    if target_in <= base_in_dim:
        raise ValueError(
            f"target in_dim={target_in} must exceed base_in_dim={base_in_dim} "
            "for zero-init concat expansion"
        )

    with torch.no_grad():
        target_w.zero_()
        target_w[:, :base_in_dim].copy_(
            ckpt_weight.to(device=target_w.device, dtype=target_w.dtype)
        )

    print(
        f"[SRDiT] patch_embedding expanded {ckpt_in}->{target_in}, "
        f"zero-init cond channels [{base_in_dim}:{target_in}]"
    )


def _load_dit_state_dict_with_patch_expand(
    model: nn.Module,
    sd: dict,
    *,
    zero_init_cond_channels: bool = False,
    base_in_dim: Optional[int] = None,
) -> tuple:
    """Load DiT weights, optionally expanding patch_embedding with zero-init cond channels."""
    sd = dict(sd)
    pe_key = "patch_embedding.weight"
    if zero_init_cond_channels and pe_key in sd:
        _apply_patch_embedding_expand_zero_init(
            model.patch_embedding, sd.pop(pe_key), base_in_dim=base_in_dim
        )
    return model.load_state_dict(sd, strict=False)


# ---------------------------------------------------------------------------
# SRDiTWrapper
# ---------------------------------------------------------------------------

class SRDiTWrapper(nn.Module):
    """Wraps the SR DiT Transformer3DModel for training.

    Interface-compatible with the training-side diffusion wrapper:
    - forward(noisy_image_or_video, conditional_dict, timestep, ...)
    - Scheduler-based flow_pred → x0 conversion
    - get_scheduler() / post_init()
    - Checkpoint loading with patch_embedding expansion
    """

    def __init__(
        self,
        dit_config: dict,
        checkpoint: Optional[str] = None,
        wan_model_dir: Optional[str] = None,
        timestep_shift: float = 5.0,
    ):
        super().__init__()
        cfg = dict(dit_config)
        zero_init_cond = cfg.pop("zero_init_cond_channels", False)
        base_in_dim = cfg.pop("base_in_dim", None)

        self.stream_kv_len = cfg.pop("stream_kv_len", None)
        # stream_kv_len <= 0 (e.g. -1) means "keep the entire KV cache" (no
        # truncation). Normalize to None here so the value never reaches the
        # downstream kv_len>0 validators in the attention modules.
        if self.stream_kv_len is not None and int(self.stream_kv_len) <= 0:
            self.stream_kv_len = None
        self.model = Transformer3DModel(**cfg)
        self.num_train_timesteps = 1000
        self.uniform_timestep = True

        ps = self.model.patch_size
        self.seq_len = 32760  # default upper bound; overridden per-call

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(self.num_train_timesteps, training=True)
        self.post_init()

        ckpt_path = checkpoint or wan_model_dir
        if ckpt_path and Path(ckpt_path).exists():
            print(f"[SRDiT] Loading weights from {ckpt_path}")
            sd = _normalize_dit_state_dict(_load_dit_state_dict(ckpt_path))
            missing, unexpected = _load_dit_state_dict_with_patch_expand(
                self.model, sd,
                zero_init_cond_channels=zero_init_cond,
                base_in_dim=base_in_dim,
            )
            loaded = len(sd) - len(missing)
            print(f"[SRDiT] Loaded checkpoint: {loaded}/{len(sd)} params matched, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")
            if missing:
                print(f"[SRDiT] Missing keys (first 5): {missing[:5]}")
            if unexpected:
                print(f"[SRDiT] Unexpected keys (first 5): {unexpected[:5]}")
            del sd
        elif ckpt_path:
            print(f"[SRDiT] Warning: checkpoint path not found ({ckpt_path}), using random init")
        else:
            print("[SRDiT] No checkpoint specified, using random initialization")

    def enable_gradient_checkpointing(self) -> None:
        for module in list(self.model.modules()):
            if hasattr(module, "_set_gradient_checkpointing"):
                module._set_gradient_checkpointing(enable=True)

    @property
    def patch_size(self):
        return self.model.patch_size

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()

    # ------------------------------------------------------------------
    # Flow <-> x0 conversion (scheduler-based)
    # ------------------------------------------------------------------

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt = flow_pred.double(), xt.double()
        sigma_t = (timestep.double().to(flow_pred.device) / self.num_train_timesteps).reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = x0_pred.dtype
        x0_pred, xt = x0_pred.double(), xt.double()
        sigma_t = (timestep.double().to(x0_pred.device) / scheduler.num_train_timesteps).reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_caches: Optional[List] = None,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cond_y: Optional[torch.Tensor] = None,
        is_stream: bool = False,
        anchor_kv: Optional[List] = None,
        anchor_cfg: Optional[dict] = None,
        **model_kwargs,
    ):
        """
        Args:
            noisy_image_or_video: [B, T, C, H, W]
            conditional_dict: {"prompt_embeds": [B, L, D]}
            timestep: [B] or [B, T]
            kv_caches: per-layer KV caches for streaming inference
            clean_x: [B, T, C, H, W] for teacher forcing
            aug_t: [B] timestep for clean GT in teacher forcing
            cond_y: [B, T, C, H, W] channel concat condition (SR-specific)
            is_stream: streaming inference flag
            **model_kwargs: pass-through to WanModel.forward (kv_len, temporal_offset, ...)

        Returns:
            Training: (flow_pred, pred_x0)  both [B, T, C, H, W]
            Streaming: (flow_pred, pred_x0, new_kv_caches)
        """
        prompt_embeds = conditional_dict["prompt_embeds"]

        b, t_frames, c, h, w = noisy_image_or_video.shape

        if self.uniform_timestep and timestep.dim() == 2:
            t_input = timestep[:, 0].float()
        elif timestep.dim() == 2:
            t_input = timestep.float()
        else:
            t_input = timestep.float()

        x_list = [noisy_image_or_video[i].permute(1, 0, 2, 3) for i in range(b)]
        y_list = [cond_y[i].permute(1, 0, 2, 3) for i in range(b)] if cond_y is not None else None

        clean_list = None
        if clean_x is not None:
            clean_list = [clean_x[i].permute(1, 0, 2, 3) for i in range(b)]

        if prompt_embeds.dim() == 3:
            ctx_list = [prompt_embeds[i] for i in range(b)]
        else:
            ctx_list = prompt_embeds

        seq_len = t_frames * h * w // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

        if clean_x is not None and aug_t is None:
            aug_t = torch.zeros_like(t_input)

        if is_stream and self.stream_kv_len is not None and "kv_len" not in model_kwargs:
            model_kwargs["kv_len"] = self.stream_kv_len

        # LQ-anchor pass-through (streaming only). None -> unchanged behaviour.
        if anchor_kv is not None:
            model_kwargs["anchor_kv"] = anchor_kv
            model_kwargs["anchor_cfg"] = anchor_cfg

        model_out = self.model(
            x=x_list, t=t_input, context=ctx_list,
            seq_len=seq_len, y=y_list,
            clean_x=clean_list, aug_t=aug_t,
            kv_caches=kv_caches, is_stream=is_stream,
            **model_kwargs,
        )

        if is_stream or kv_caches is not None:
            outputs, new_kv_caches = model_out
        else:
            outputs = model_out
            new_kv_caches = None

        flow_pred = torch.stack(outputs, dim=0).permute(0, 2, 1, 3, 4)

        flat_timestep = timestep.flatten(0, 1) if timestep.dim() == 2 else timestep.repeat_interleave(t_frames)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=flat_timestep,
        ).unflatten(0, (b, t_frames))

        if new_kv_caches is not None:
            return flow_pred, pred_x0, new_kv_caches

        return flow_pred, pred_x0
