"""Main pipeline module for continuous GSLM.

This module contains the GSLMPipeline class which orchestrates the entire
model including SSL encoders, decoders, and all processing steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from speechbrain.dataio.dataio import length_to_mask

from encoders import (
    MimiEncoder, 
)
from decoders import (
    ELMDecoderWrapper,
)
from model_utils import reduce_features, split_features


class GSLMPipeline(nn.Module):
    """Main pipeline for continuous GSLM.
    
    This class orchestrates the entire model including SSL encoders, decoders,
    and all processing steps from raw audio to final outputs.
    """
    
    def __init__(self, conf, args):
        super().__init__()
        self.conf = conf
        self.args = args
        
        if hasattr(self.conf.model, "ssl_model") and self.conf.model.ssl_model == "mimi":
            n_quantizers = 0 if not hasattr(self.conf.model, "n_quantizers") else self.conf.model.n_quantizers
            self.ssl_model = MimiEncoder(freeze=self.conf.model.freeze, n_quantizers=n_quantizers)

        # Initialize decoder model
        if self.conf.model.decoder == "OpenELM-270M":
            from transformers import AutoModelForCausalLM
            attn_implementation = "flash_attention_2" if self.conf.model.flash_attention else "eager"
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or attn_implementation == 'flash_attention_2' else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                "apple/OpenELM-270M", 
                torch_dtype=torch_dtype, 
                trust_remote_code=True
            )
        elif self.conf.model.decoder == "OpenELM-450M":
            from transformers import AutoModelForCausalLM
            attn_implementation = "flash_attention_2" if self.conf.model.flash_attention else "eager"
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or attn_implementation == 'flash_attention_2' else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                "apple/OpenELM-450M", 
                torch_dtype=torch_dtype, 
                trust_remote_code=True
            )
        elif self.conf.model.decoder == "OpenELM-1B":
            from transformers import AutoModelForCausalLM
            attn_implementation = "flash_attention_2" if self.conf.model.flash_attention else "eager"
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or attn_implementation == 'flash_attention_2' else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                "apple/OpenELM-1_1B", 
                torch_dtype=torch_dtype, 
                trust_remote_code=True
            )
        elif self.conf.model.decoder == "OpenELM-3B":
            from transformers import AutoModelForCausalLM
            attn_implementation = "flash_attention_2" if self.conf.model.flash_attention else "eager"
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or attn_implementation == 'flash_attention_2' else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                "apple/OpenELM-3B", 
                torch_dtype=torch_dtype, 
                trust_remote_code=True
            )
            
        ssl_dim, reduction_factor = self.conf.model.ssl_dim, self.conf.model.reduction_factor

        # Initialize normalization
        if hasattr(self.conf.model, "norm") and self.conf.model.norm == "static":
            mean = np.load(self.conf.model.mean_path)
            self.register_buffer('mean', torch.from_numpy(mean).float())
            std = np.load(self.conf.model.std_path)
            self.register_buffer('std', torch.from_numpy(std).float())

        if self.conf.optimizer.loss_function == "FM":
            input_dim = ssl_dim * reduction_factor
            output_dim = ssl_dim * reduction_factor
        else:
            raise NotImplementedError(f"Loss function {self.conf.optimizer.loss_function} not supported.")

        # Initialize auxiliary output dimensions for token prediction
        if self.conf.model.ssl_model == "mimi" and (self.conf.optimizer.token_loss_weight > 0 or self.conf.model.token_conditioning):
            aux_output_dim = self.ssl_model.model.config.codebook_size * reduction_factor * self.conf.model.n_quantizers
            # used as eos
            if hasattr(self.conf.model, "n_special_tokens") and self.conf.model.n_special_tokens > 0:
                aux_output_dim += self.conf.model.n_special_tokens
                self.eos_token_index = aux_output_dim - 1
            if hasattr(self.conf.model, "extra_future_tokens") and self.conf.model.extra_future_tokens > 0:
                aux_output_dim *= self.conf.model.extra_future_tokens
        else:
            aux_output_dim = None

        # Initialize token embedding dimensions
        token_emb_dim = self.conf.model.token_emb_dim if hasattr(self.conf.model, "token_emb_dim") and hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning else 0
        if hasattr(self.conf.model, "future_conditioning") and self.conf.model.future_conditioning:
            token_emb_dim *= self.conf.model.extra_future_tokens

        if "OpenELM" in self.conf.model.decoder:
            output_layer = "simple_mlp" if self.conf.optimizer.loss_function == "FM" else "linear" 

            n_res_blocks = self.conf.model.n_res_blocks
            if hasattr(self.conf.model, "ssl_model") and self.conf.model.ssl_model == "mimi":
                # for mimi use the newer OPTDecoderWrapperV2
                self.decoder = ELMDecoderWrapper(
                    model,
                    input_dim=input_dim,
                    decoder_dim=self.conf.model.decoder_dim,
                    output_dim=output_dim,
                    aux_output_dim=aux_output_dim,
                    output_layer=output_layer,
                    n_res_blocks=n_res_blocks,
                    aux_output_layer_idx=None if not hasattr(self.conf.model, "aux_output_layer_idx") else self.conf.model.aux_output_layer_idx,
                    token_emb_dim=token_emb_dim,
                )
            self.pad_index = model.config.pad_token_id 
            self.bos_index = model.config.bos_token_id 
            self.eos_index = model.config.eos_token_id 

        # Initialize embeddings
        self.embed = nn.Embedding(3, embedding_dim=input_dim, padding_idx=self.pad_index)
        if hasattr(self.conf.model, "mask_vector") and self.conf.model.mask_vector:
            self.mask_embed = nn.Embedding(2, embedding_dim=ssl_dim)

        nn.init.normal_(self.embed.weight, mean=0, std=input_dim ** -0.5)
        nn.init.constant_(self.embed.weight[self.pad_index], 0)
        
        # Initialize token embeddings if needed
        if (hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning) or (hasattr(self.conf.model, "token_input") and self.conf.model.token_input):
            # add token emb to z, only support mimi
            if hasattr(self.conf.model, "add_special_token_to_embedding_table") and self.conf.model.add_special_token_to_embedding_table:
                self.token_embed = nn.Embedding(self.ssl_model.model.config.codebook_size + self.conf.model.n_special_tokens, embedding_dim=self.conf.model.token_emb_dim)
            else:
                self.token_embed = nn.Embedding(self.ssl_model.model.config.codebook_size, embedding_dim=self.conf.model.token_emb_dim) 

    def forward(self, wavs, wav_len):
        with torch.no_grad():
            if self.conf.model.ssl_model == "mimi" and hasattr(self.conf.model, "n_quantizers") and self.conf.model.n_quantizers > 0:
                ssl_feats, tokens = self.ssl_model(wavs, wav_len, layer_idx=self.conf.model.layer_idx)
            else:
                ssl_feats = self.ssl_model(wavs, wav_len, layer_idx=self.conf.model.layer_idx)

            ssl_abs_len = torch.round(wav_len * ssl_feats.shape[1]).long()
            ssl_padding_mask = ~length_to_mask(ssl_abs_len, dtype=torch.bool)
            
            if hasattr(self.conf.model, "norm") and self.conf.model.norm == "static":
                ssl_feats = (ssl_feats - self.mean) / self.std

        # Reduce features
        if self.conf.model.reduction_factor > 1:
            reduced_ssl_feats = reduce_features(ssl_feats, self.conf.model.reduction_factor, pad=self.conf.model.pad_feature)
        else:
            reduced_ssl_feats = ssl_feats

        # Prepending BOS token
        bs = reduced_ssl_feats.shape[0]
        bos_token = ssl_feats.new_ones((bs, 1)).long() * self.bos_index
        bos_vec = self.embed(bos_token)

        # Prepare input tokens based on configuration
        if hasattr(self.conf.model, "token_input") and self.conf.model.token_input:
            token_embs = self.token_embed(tokens[:, :, 0])
            prev_tokens = torch.cat([bos_vec, token_embs], dim=1)
        else:
            # use continuous input
            prev_tokens = torch.cat([bos_vec, reduced_ssl_feats], dim=1)

        # Forward through decoder
        abs_len = torch.round(wav_len * reduced_ssl_feats.shape[1]).long() + 1
        padding_mask = length_to_mask(abs_len, max_len=prev_tokens.shape[1], dtype=torch.bool)
        logits, aux_output = self.decoder(prev_tokens, padding_mask)

        # Remove the last frame because there is no loss applied on it
        logits = logits[:, :-1]

        # Process token predictions
        if self.conf.model.ssl_model == "mimi" and (self.conf.optimizer.token_loss_weight > 0 or self.conf.model.token_conditioning) and self.conf.model.n_quantizers > 0:
            if hasattr(self.conf.model, "n_special_tokens") and self.conf.model.n_special_tokens > 0:
                token_logits = split_features(aux_output, self.conf.model.reduction_factor * self.conf.model.n_quantizers)
                k = 1 if not hasattr(self.conf.model, "extra_future_tokens") or self.conf.model.extra_future_tokens == 0 else self.conf.model.extra_future_tokens
                ssl_abs_len = torch.round(wav_len * tokens.shape[1]).long()
                split_padding_mask = length_to_mask(ssl_abs_len + 1, dtype=torch.bool)
                # append eos as last token
                eos_index = self.eos_token_index
                tokens = torch.cat([tokens, tokens.new_ones((bs, k, 1)).long() * eos_index], dim=1)
                B = tokens.shape[0]
                offsets = ssl_abs_len.unsqueeze(1) + torch.arange(k, device=tokens.device).unsqueeze(0)  # shape [B, k]
                batch_indices = torch.arange(B, device=tokens.device).unsqueeze(1).expand(B, k)         # shape [B, k]
                tokens[batch_indices, offsets] = eos_index
        else:
            token_logits = None
            tokens = None
            split_padding_mask = None

        # Apply token conditioning if specified
        if hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning:
            L = logits.shape[1]
            if not hasattr(self.conf.model, "future_conditioning") or not self.conf.model.future_conditioning:
                # use only the first token
                token_embed = self.token_embed(tokens[:, :L, 0])
                logits = torch.cat([logits, token_embed], dim=2) 
            else:
                k = 1 if self.conf.model.extra_future_tokens == 0 else self.conf.model.extra_future_tokens
                conditioning_tokens = torch.stack([tokens[:, kk:kk+L, 0] for kk in range(k)], dim=2) # [B, T, k]
                token_embed = self.token_embed(conditioning_tokens).flatten(start_dim=2, end_dim=-1) # [B, T, k * D]
                logits = torch.cat([logits, token_embed], dim=2)

        # Remove one frame for loss computation
        padding_mask_for_loss = padding_mask.clone()
        padding_mask_for_loss[torch.arange(bs, device=padding_mask_for_loss.device), abs_len - 1] = 0
        padding_mask_for_loss = padding_mask_for_loss[:, :-1].unsqueeze(dim=2)

        return logits, reduced_ssl_feats, padding_mask_for_loss, token_logits, tokens, split_padding_mask