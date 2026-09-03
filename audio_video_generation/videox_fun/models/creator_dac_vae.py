import os
from typing import Optional
from pathlib import Path

import torch

from videox_fun.models.creator.dac_vae import DAC, DiagonalGaussianDistribution


class CreatorDACVAE(torch.nn.Module):
    """
    High-level wrapper around the DAC (Descript Audio Codec) VAE in continuous mode.
    Mirrors the LTXAudioVAE interface used by the inference pipeline.
    """

    def __init__(self, dac_model: DAC) -> None:
        super().__init__()
        self.dac = dac_model

    @property
    def sample_rate(self) -> int:
        return self.dac.sample_rate

    @property
    def hop_length(self) -> int:
        return self.dac.hop_length

    @property
    def latent_dim(self) -> int:
        return self.dac.latent_dim

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str | os.PathLike[str],
        strict: bool = False,
    ) -> "CreatorDACVAE":
        pretrained_model_path = Path(pretrained_model_path)

        if pretrained_model_path.is_dir():
            dac_model = DAC.from_pretrained(pretrained_model_path)
        else:
            dac_model = DAC.from_pretrained(pretrained_model_path.parent)

        return cls(dac_model=dac_model)

    def _preprocess_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform to [B, 1, T] mono and pad to hop_length boundary."""
        """The Creator audio VAE currently supports mono audio."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            if waveform.size(1) > 1:
                waveform = waveform.mean(dim=1, keepdim=True)

        waveform = self.dac.preprocess(waveform, self.sample_rate)
        return waveform

    @torch.inference_mode()
    def encode_posterior(
        self,
        audio: torch.Tensor,
        sampling_rate: Optional[int] = None,
        deterministic: bool | None = None,
    ) -> DiagonalGaussianDistribution:
        """Encode audio waveform and return the posterior distribution.

        Parameters
        ----------
        audio : Tensor
            Raw waveform tensor. Accepts shapes [T], [C, T], or [B, C, T].
        sampling_rate : int, optional
            Not used directly; kept for API compatibility with LTXAudioVAE.
        deterministic : bool, optional
            If True, std is zeroed so sampling returns the mean.

        Returns
        -------
        DiagonalGaussianDistribution
        """
        waveform = self._preprocess_waveform(audio)
        posterior, _, _, _, _ = self.dac.encode(waveform)
        if deterministic is not None:
            posterior.deterministic = deterministic
            if deterministic:
                posterior.std = torch.zeros_like(posterior.mean)
                posterior.var = torch.zeros_like(posterior.mean)
        return posterior

    @torch.inference_mode()
    def encode(
        self,
        audio: torch.Tensor,
        sampling_rate: Optional[int] = None,
        sample: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Encode audio waveform to latent representation.

        Parameters
        ----------
        audio : Tensor
            Raw waveform tensor.
        sampling_rate : int, optional
            Kept for API compatibility.
        sample : bool
            If True, sample from the posterior; otherwise return the mean.
        generator : torch.Generator, optional
            RNG for reproducible sampling.

        Returns
        -------
        Tensor [B, D, T']
            Continuous latent codes.
        """
        posterior = self.encode_posterior(audio, sampling_rate=sampling_rate)
        if sample:
            return posterior.sample()
        return posterior.mode()

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent codes back to waveform.

        Parameters
        ----------
        latent : Tensor [B, D, T']
            Continuous latent codes.

        Returns
        -------
        Tensor [B, 1, T]
            Reconstructed waveform.
        """
        return self.dac.decode(latent)

    @torch.inference_mode()
    def reconstruct(
        self,
        audio: torch.Tensor,
        sampling_rate: Optional[int] = None,
        sample: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Encode then decode (round-trip reconstruction)."""
        latent = self.encode(audio, sampling_rate=sampling_rate, sample=sample, generator=generator)
        return self.decode(latent)
