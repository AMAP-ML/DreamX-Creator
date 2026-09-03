"""Load ``wan/modules/vae2_2.py`` by file path.

``import wan.modules.vae2_2`` goes through ``wan/__init__.py`` → ``t5.py``,
which calls ``torch.cuda.current_device()`` at class-body time. That grabs a
CUDA context (and crashes if ``CUDA_VISIBLE_DEVICES`` is empty). This package
only needs the VAE building blocks, so we exec the file directly.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_VAE22_PATH = Path(__file__).resolve().parent.parent / "wan" / "modules" / "vae2_2.py"
_MOD = None


def wan_vae22():
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location(
            "lightvae_nu._wan_vae2_2", _VAE22_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD = mod
    return _MOD
