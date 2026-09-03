"""Single-sample Creator inference with Wan2.2-style Ulysses SP."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from torchvision.io import write_video

import inference as creator_inference


def parse_args():
    # Remove the SP-only option before reusing the release inference parser.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sp_size", type=int, default=None)
    parallel_args, remaining = parser.parse_known_args()
    import sys

    sys.argv = [sys.argv[0], *remaining]
    args = creator_inference.parse_args()
    args.sp_size = parallel_args.sp_size
    return args


def _save_outputs(args, models, video_decoded, audio_decoded):
    output_path = Path(args.output)
    video_path = output_path.with_suffix(".video.mp4")
    audio_path = output_path.with_suffix(".wav")
    frames = (
        video_decoded[0].permute(1, 2, 3, 0).clamp(0, 1).numpy() * 255
    ).astype("uint8")
    write_video(
        str(video_path),
        torch.from_numpy(frames),
        fps=args.fps,
        video_codec="h264",
    )
    creator_inference.save_audio_wav(
        audio_decoded,
        int(models["audio_vae"].sample_rate),
        str(audio_path),
    )
    mux_result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if mux_result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while muxing {output_path}:\n{mux_result.stderr.strip()}"
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SP inference requires CUDA")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size < 2:
        raise RuntimeError("Launch SP inference with torchrun and at least two GPUs")
    if args.sp_size is not None and args.sp_size != world_size:
        raise ValueError(f"--sp_size={args.sp_size} must equal WORLD_SIZE={world_size}")
    if rank != 0:
        creator_inference.print_info = lambda *args, **kwargs: None

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://")

    output_path = Path(args.output).resolve()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(output_path)
    args.output_dir = str(output_path.parent)
    args.disable_progress = rank != 0
    args.suppress_aux_writes = rank != 0
    args.synchronize_noise = True
    args.skip_output_decode = rank != 0

    weight_dtype = creator_inference.resolve_weight_dtype(args.weight_dtype, device)
    models = creator_inference.setup_models(args, device, weight_dtype)
    transformer = models["transformer"]
    base_transformer = getattr(transformer, "model", transformer)
    if not hasattr(base_transformer, "enable_multi_gpus_inference"):
        raise RuntimeError("The loaded Creator transformer does not support SP inference")
    base_transformer.enable_multi_gpus_inference()

    item = {
        "prompt": args.prompt,
        "video_prompt": args.prompt,
        "audio_prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "audio_negative_prompt": args.negative_prompt,
        "duration": args.duration,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "name": "sample",
    }
    video_decoded, audio_decoded, _ = creator_inference.generate_joint_audio_video(
        args, models, device, weight_dtype, item
    )

    dist.barrier()
    if rank == 0:
        _save_outputs(args, models, video_decoded, audio_decoded)
        creator_inference.print_info(f"Saved: {output_path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
