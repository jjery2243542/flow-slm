"""Encoder components for speech representation learning.

This module contains various encoder implementations including WavLM, Mimi,
and Fairseq-based encoders for extracting speech representations.
"""

import torch
import torch.nn as nn
import torchaudio
import logging
from typing import Optional, Tuple, Union
from contextlib import nullcontext

from transformers import MimiModel, AutoFeatureExtractor

logger = logging.getLogger(__name__)


class MimiEncoder(torch.nn.Module):
    """Mimi encoder for speech representation learning."""
    
    def __init__(self, freeze: bool = True, n_quantizers: int = 0):
        super().__init__()
        self.model = MimiModel.from_pretrained("kyutai/mimi")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        self.resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)
        self.freeze = freeze
        self.n_quantizers = n_quantizers
        
        if freeze:
            self.model.eval()
            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract Mimi features from input waveform.
        
        Args:
            wavs: Input waveform tensor
            wav_lens: Waveform length tensor
            
        Returns:
            Extracted features tensor, optionally with quantized codes
        """
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            resampled_wavs = self.resample(wavs)
            embeddings = self.model.encoder(resampled_wavs.unsqueeze(dim=1))
            encoder_outputs = self.model.encoder_transformer(
                embeddings.transpose(1, 2), past_key_values=None, return_dict=None
            )
            embeddings = encoder_outputs[0].transpose(1, 2)
            embeddings = self.model.downsample(embeddings)

        if self.n_quantizers > 0:
            codes = self.model.quantizer.encode(embeddings, self.n_quantizers)
            codes = codes.transpose(0, 1)
            return embeddings.transpose(1, 2), codes.transpose(1, 2)  # [B, T, F], [B, T, C]
        else:
            return embeddings.transpose(1, 2)


class MimiDecoder(torch.nn.Module):
    """Mimi decoder for speech synthesis."""
    
    def __init__(self):
        super().__init__()
        self.model = MimiModel.from_pretrained("kyutai/mimi")

    def forward(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None, return_codes: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode Mimi embeddings to audio.
        
        Args:
            embeddings: Input embeddings tensor
            num_quantizers: Number of quantizers to use
            return_codes: Whether to return quantized codes
            
        Returns:
            Decoded audio tensor, optionally with codes
        """
        num_quantizers = self.model.config.num_quantizers if num_quantizers is None else num_quantizers
        embeddings = embeddings.transpose(1, 2)
        codes = self.model.quantizer.encode(embeddings, num_quantizers)
        codes = codes.transpose(0, 1)
        audio_values = self.model.decode(codes)[0].squeeze(dim=1)
        if not return_codes:
            return audio_values
        else:
            return audio_values, codes
