<div align="center">
  <img src="dreamx-creator_teaser.png" alt="DreamX-Creator teaser">

<h1>DreamX-Creator 1.0: Democratizing Native Audio-Video Generation</h1>

DreamX Team

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2608.31106-b31b1b.svg)](https://arxiv.org/abs/2608.31106)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

</div>

-----

**DreamX-Creator 1.0** is a research framework for **native joint audio-video generation**. Given a first frame and a text prompt, its implemented base generator jointly models modality-specialized video and audio streams, using **Gated Cross-Modal Attention** and **Progressive Joint Training** to enable bidirectional audio-video interaction.

The broader system combines **Audio-Video Reinforcement Learning** with **Modality-Aware Multimodal Feedback** to improve visual and audio quality, semantic consistency, and fine-grained audio-video synchronization. **Autoregressive 1-Step 2K Refinement** then upgrades the generated video to high-quality 2K output while preserving content, motion, and audio-aligned timing.

## :fire: News

- **Sep 1, 2026:** Initialized the DreamX-Creator project repository with its overview and release roadmap.

## :calendar: Plan

- :heavy_check_mark: Initialize the DreamX-Creator project repository.
- [ ] Release the DreamX-Creator 1.0 technical report.
- [ ] Release validated model weights, inference code, configurations, and evaluation tools.

## :books: Citation

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

## :scroll: License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## :sparkles: Acknowledgement

We would like to thank the [Wan Team](https://github.com/Wan-Video/Wan2.2) and the [OpenMOSS Team](https://github.com/OpenMOSS/MOVA) for their outstanding open-source work on Wan and MOVA, respectively.
