"""Model definitions required by the release inference entrypoint."""

from .creator_dac_vae import CreatorDACVAE
from .wan_text_encoder import WanT5EncoderModel
from .wan_vae3_8 import AutoencoderKLWan3_8

__all__ = ["AutoencoderKLWan3_8", "CreatorDACVAE", "WanT5EncoderModel"]
