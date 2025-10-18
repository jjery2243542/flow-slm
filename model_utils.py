"""Utility functions for model operations.

This module contains utility functions used across different model components
including masking, feature processing, and attention mask generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from fairseq.data.data_utils import compute_mask_indices


def apply_mask(
    input: torch.Tensor, 
    padding_mask: Optional[torch.Tensor], 
    mask_prob: float = 0.65, 
    mask_length: int = 10, 
    learned_vector: Optional[torch.Tensor] = None, 
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, np.ndarray]:
    """Apply mask to input tensor.
    
    Args:
        input: Input tensor to mask
        padding_mask: Padding mask for the input
        mask_prob: Probability of masking
        mask_length: Length of each mask
        learned_vector: Learned vector to replace masked positions
        device: Device to use for computations
        
    Returns:
        Tuple of (masked_input, mask_indices)
    """
    # Don't use the same masks because the lengths are not the same
    mask_indices = compute_mask_indices(
        input.shape[:-1], 
        padding_mask=padding_mask, 
        mask_prob=mask_prob, 
        mask_length=mask_length, 
        require_same_masks=False
    )
    masked_input = input.clone()
    if learned_vector is None:
        masked_input[mask_indices] = 0
    else:
        masked_input[mask_indices] = learned_vector
    return masked_input, mask_indices


def reduce_features(features: torch.Tensor, reduction: int, pad: bool = False) -> torch.Tensor:
    """Reduce features by concatenating consecutive frames.
    
    Args:
        features: Input features tensor [B, T, F]
        reduction: Reduction factor
        pad: Whether to pad if sequence length is not divisible by reduction
        
    Returns:
        Reduced features tensor [B, T//reduction, F*reduction]
    """
    remain = features.shape[1] % reduction
    if not pad and remain > 0:
        features = features[:, :-remain, :]
    elif pad and remain > 0:
        # Zero-padding for now
        features = F.pad(features, pad=(0, 0, 0, reduction - remain)) 

    bs, seq_len, feat_dim = features.shape
    reduced_features = features.reshape(bs, seq_len // reduction, reduction * feat_dim)
    return reduced_features


def split_features(features: torch.Tensor, reduction: int) -> torch.Tensor:
    """Split features by expanding frames.
    
    Args:
        features: Input features tensor [B, T, F]
        reduction: Expansion factor
        
    Returns:
        Split features tensor [B, T*reduction, F//reduction]
    """
    bs, seq_len, feat_dim = features.shape
    split_features = features.reshape(bs, seq_len * reduction, feat_dim // reduction)
    return split_features


def generate_attention_mask(seq_len: int, K: int, M: int) -> torch.Tensor:
    """Generate an attention mask for a sequence.
    
    The mask allows:
    1. The first K tokens to attend to all of the K tokens
    2. Every subsequent M tokens to attend to each other and all previous tokens
    
    Args:
        seq_len: Total length of the sequence
        K: Number of initial tokens that can attend to each other
        M: Size of subsequent groups of tokens
        
    Returns:
        A (seq_len, seq_len) attention mask matrix
    """
    # Initialize the attention mask with zeros
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)

    # First K tokens can attend to each other
    mask[:K, :K] = True

    # Subsequent groups of M tokens
    for start in range(K, seq_len, M):
        end = min(start + M, seq_len)
        # Current group can attend to all previous tokens and itself
        mask[start:end, :end] = True

    return mask


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization modulation.
    
    Args:
        x: Input tensor
        shift: Shift parameter
        scale: Scale parameter
        
    Returns:
        Modulated tensor
    """
    return x * (1 + scale) + shift