<div align="center">
  <img src="./dreamx-creator_teaser.png" alt="DreamX-Creator teaser">

  <h1>DreamX-Creator 1.0: Model Weights</h1>

  DreamX Team

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2608.31106-b31b1b.svg)](https://arxiv.org/abs/2608.31106)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](../LICENSE)

</div>

-----

This directory holds all model weights for **DreamX-Creator 1.0**, a research
framework for **native joint audio-video generation**. Given a first frame and
a text prompt, the 7B base generator jointly models modality-specialized video
and audio streams; the **Autoregressive 1-Step 2K Refiner** (SR-DiT 5B) then
upgrades the generated video to high-quality 2K output.

Weights are **not stored in the git repository**. They are distributed on
[HuggingFace](https://huggingface.co/GD-ML/DreamX-Creator) and
[ModelScope](https://modelscope.cn/models/GD-ML/DreamX-Creator) and should be
placed under this directory following the layout below.

## Expected Layout

```
checkpoints/
├── creator/                     # DreamX-Creator 1.0 joint generator (7B, LoRA merged)
│   ├── video_model/             # video DiT shards + config
│   ├── audio_model/             # audio DiT + config
│   └── cross_attn_weights.safetensors  # gated A2V/V2A cross-modal attention
├── audio_vae/                   # CreatorDACVAE audio VAE
├── refiner/                     # 2K refiner
│   ├── sr_dit_5b.pt             # SR-DiT 5B refiner
│   ├── latent_upsampler_flash.pt       # FlashLatentUpsampler (default)
│   ├── latent_upsampler_2d_causal.pt   # causal 2D latent upsampler (optional)
│   └── lightvae_nu_scheme3.pt          # distilled fast decoder (optional, off by default)
└── wan2.2_ti2v_5b/              # shared Wan2.2-TI2V-5B dependencies
    ├── Wan2.2_VAE.pth           # video VAE
    ├── models_t5_umt5-xxl-enc-bf16.pth  # UMT5-xxl text encoder
    └── google/umt5-xxl/         # tokenizer
```

The `wan2.2_ti2v_5b/` directory can also be downloaded directly from
[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B); only the
three entries above are needed. It is shared by both pipelines.

## Which Weights Are Used Where

- **Joint audio-video generation** uses `creator/`, `audio_vae/`, and
  `wan2.2_ti2v_5b/`. See the
  [audio_video_generation README](../audio_video_generation/README.md) for
  setup, input overrides, and memory options.
- **2K refinement** uses `refiner/` and `wan2.2_ti2v_5b/`. See the
  [video_refiner README](../video_refiner/README.md) for setup and the full
  list of inference knobs.

## Quickstart

Once the weights above are in place, from the repository root:

**1. Joint audio-video generation** (first frame + prompt to synchronized
video with audio):

```bash
cd audio_video_generation
pip install -r requirements.txt
./inference.sh                        # runs the default Verse-Bench case (case1)
```

**2. 2K refinement** (super-resolve a generated or external video, audio
unchanged):

```bash
cd ../video_refiner
pip install -r requirements.txt
INPUT=/path/to/video.mp4 bash run_inference.sh
```

## Citation

If you find DreamX-Creator useful in your research, please consider citing our technical report:

```bibtex
@misc{zhu2026dreamxcreatordemocratizingnativeaudiovideo,
  title={DreamX-Creator: Democratizing Native Audio-Video Generation at 2K Resolution},
  author={Jiashu Zhu and Yanhao Zheng and Ruitian Tian and Rujing Dang and Shen Zhang and Bingze Song and Jiachen Lei and Ruimin Lin and Jiahong Wu and Xiangxiang Chu},
  year={2026},
  eprint={2608.31106},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.31106},
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Acknowledgement

We would like to thank the [Wan Team](https://github.com/Wan-Video/Wan2.2), the [OpenMOSS Team](https://github.com/OpenMOSS/MOVA), and the [VideoX-Fun Team](https://github.com/aigc-apps/VideoX-Fun) for their outstanding open-source work on Wan, MOVA, and VideoX-Fun, respectively.
