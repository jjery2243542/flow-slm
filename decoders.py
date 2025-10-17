"""Decoder wrapper components for language modeling.

This module contains various decoder wrapper implementations for different
language models including OPT, OpenELM, and Fairseq-based decoders.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers import AutoModelForCausalLM
from diffusion import SimpleMLPAdaLN


class BaseDecoderWrapper(torch.nn.Module):
    """Base class for decoder wrappers."""
    
    def __init__(
        self, 
        model, 
        input_dim: int, 
        decoder_dim: int, 
        output_dim: int, 
        aux_output_dim: Optional[int] = None, 
        output_layer: str = "linear", 
        n_res_blocks: int = 3, 
        aux_output_layer_idx: Optional[int] = None, 
        token_emb_dim: int = 0
    ):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, decoder_dim)
        self.aux_output_layer_idx = aux_output_layer_idx
        self.output_layer_type = output_layer
        self.frozen = False

        # Initialize output projection
        if output_layer == "linear":
            self.output_proj = torch.nn.Linear(decoder_dim, output_dim)
        elif output_layer == "simple_mlp":
            if decoder_dim > 1280:
                self.output_proj = SimpleMLPAdaLN(output_dim, decoder_dim, output_dim, decoder_dim + token_emb_dim, n_res_blocks)
            else:
                self.output_proj = SimpleMLPAdaLN(output_dim, decoder_dim * 2, output_dim, decoder_dim + token_emb_dim, n_res_blocks)

        if aux_output_dim:
            self.aux_output_proj = torch.nn.Linear(decoder_dim, aux_output_dim)


class ELMDecoderWrapper(BaseDecoderWrapper):
    """Decoder wrapper for OpenELM models."""
    
    def __init__(
        self, 
        elm, 
        input_dim: int, 
        decoder_dim: int, 
        output_dim: int, 
        aux_output_dim: Optional[int] = None, 
        output_layer: str = "linear", 
        n_res_blocks: int = 3, 
        aux_output_layer_idx: Optional[int] = None, 
        token_emb_dim: int = 0
    ):
        super().__init__(elm, input_dim, decoder_dim, output_dim, aux_output_dim, output_layer, n_res_blocks, aux_output_layer_idx, token_emb_dim)
        self.decoder = elm.transformer

    def forward(
        self,
        input_tokens: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """Forward pass for ELM decoder."""
        inputs_embeds = self.input_proj(input_tokens)
        past_seen_tokens = 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        position_ids = cache_position.unsqueeze(0)
        causal_mask = self.decoder._update_causal_mask(attention_mask, inputs_embeds)
        
        # embed positions
        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(self.decoder.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
                cache_position=cache_position,
            )
            if self.aux_output_layer_idx is not None and idx == self.aux_output_layer_idx - 1:
                aux_hidden_states = layer_outputs[0]

            hidden_states = layer_outputs[0]

        if self.aux_output_layer_idx is None:
            aux_hidden_states = hidden_states

        hidden_states = self.decoder.norm(hidden_states)

        if self.output_layer_type == "simple_mlp":
            # return hidden_states instead for later loss computation
            logits = hidden_states 
        elif self.output_layer_type == "linear":
            logits = self.output_proj(hidden_states)
        else:
            raise ValueError(f"output_layer {self.output_layer_type} not supported")

        if hasattr(self, "aux_output_proj"):
            aux_output = self.aux_output_proj(aux_hidden_states)
        else:
            aux_output = None

        # remove the last frame
        return logits, aux_output


class OPTDecoderWrapperV2(BaseDecoderWrapper):
    """Decoder wrapper for OPT models (version 2)."""
    
    def __init__(
        self, 
        opt, 
        input_dim: int, 
        decoder_dim: int, 
        output_dim: int, 
        aux_output_dim: Optional[int] = None, 
        output_layer: str = "linear", 
        n_res_blocks: int = 3, 
        aux_output_layer_idx: Optional[int] = None, 
        token_emb_dim: int = 0
    ):
        super().__init__(opt, input_dim, decoder_dim, output_dim, aux_output_dim, output_layer, n_res_blocks, aux_output_layer_idx, token_emb_dim)
        self.decoder = opt.model.decoder

    def forward(
        self,
        input_tokens: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """Forward pass for OPT decoder v2."""
        inputs_embeds = self.input_proj(input_tokens)
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if attention_mask is None:
            seq_length = past_seen_tokens + inputs_embeds.shape[1]
            attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)

        causal_mask = self.decoder._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, False
        )

        # embed positions
        position_ids = torch.cumsum(attention_mask, dim=1)
        position_ids = (position_ids * attention_mask - 1).long()
        # cut positions if `past_seen_tokens` is > 0
        position_ids = position_ids[:, past_seen_tokens:]

        pos_embeds = self.decoder.embed_positions(attention_mask, past_seen_tokens, position_ids=position_ids)

        hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)

        # decoder layers
        for idx, decoder_layer in enumerate(self.decoder.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.decoder.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                layer_head_mask=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=None,
                cache_position=cache_position,
            )
            if self.aux_output_layer_idx is not None and idx == self.aux_output_layer_idx - 1:
                aux_hidden_states = layer_outputs[0]

            hidden_states = layer_outputs[0]

        if self.aux_output_layer_idx is None:
            aux_hidden_states = hidden_states

        if self.decoder.final_layer_norm is not None:
            hidden_states = self.decoder.final_layer_norm(hidden_states)

        if self.output_layer_type == "simple_mlp":
            # return hidden_states instead for later loss computation
            logits = hidden_states
        elif self.output_layer_type == "linear":
            logits = self.output_proj(hidden_states)
        else:
            raise ValueError(f"output_layer {self.output_layer_type} not supported")

        if hasattr(self, "aux_output_proj"):
            aux_output = self.aux_output_proj(aux_hidden_states)
        else:
            aux_output = None

        return logits, aux_output


class OPTDecoderWrapper(BaseDecoderWrapper):
    """Decoder wrapper for OPT models (version 1)."""
    
    def __init__(
        self, 
        opt, 
        input_dim: int, 
        decoder_dim: int, 
        output_dim: int, 
        aux_output_dim: Optional[int] = None, 
        output_layer: str = "linear", 
        n_res_blocks: int = 3, 
        aux_output_layer_idx: Optional[int] = None, 
        token_emb_dim: int = 0
    ):
        super().__init__(opt, input_dim, decoder_dim, output_dim, aux_output_dim, output_layer, n_res_blocks, aux_output_layer_idx, token_emb_dim)
        self.decoder = opt.model.decoder

    def forward(
        self,
        input_tokens: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """Forward pass for OPT decoder v1."""
        inputs_embeds = self.input_proj(input_tokens)
        use_cache = use_cache if use_cache is not None else self.decoder.config.use_cache

        input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self.decoder._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.decoder.embed_positions(attention_mask, past_key_values_length)

        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        for idx, decoder_layer in enumerate(self.decoder.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.decoder.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                layer_head_mask=None,
                past_key_value=past_key_value,
                output_attentions=None,
                use_cache=use_cache,
            )
            if self.aux_output_layer_idx is not None and idx == self.aux_output_layer_idx - 1:
                aux_hidden_states = layer_outputs[0]

            hidden_states = layer_outputs[0]

        if self.aux_output_layer_idx is None:
            aux_hidden_states = hidden_states

        if self.decoder.final_layer_norm is not None:
            hidden_states = self.decoder.final_layer_norm(hidden_states)

        if self.output_layer_type == "simple_mlp":
            # return hidden_states instead for later loss computation
            logits = hidden_states
        elif self.output_layer_type == "linear":
            logits = self.output_proj(hidden_states)
        else:
            raise ValueError(f"output_layer {self.output_layer_type} not supported")

        if hasattr(self, "aux_output_proj"):
            aux_output = self.aux_output_proj(aux_hidden_states)
        else:
            aux_output = None

        return logits, aux_output


class TransformerDecoderWrapper(BaseDecoderWrapper):
    """Decoder wrapper for Fairseq transformer models."""
    
    def __init__(
        self, 
        gslm, 
        input_dim: int, 
        decoder_dim: int, 
        output_dim: int, 
        aux_output_dim: Optional[int] = None, 
        output_layer: str = "simple_mlp", 
        n_res_blocks: int = 3, 
        token_emb_dim: int = 0
    ):
        super().__init__(gslm, input_dim, decoder_dim, output_dim, aux_output_dim, output_layer, n_res_blocks, None, token_emb_dim)
        self.decoder = gslm.decoder

    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        mask: torch.Tensor,
        encoder_out: Optional[Dict[str, List[torch.Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """Forward pass for Fairseq transformer decoder."""
        prev_output_tokens = self.input_proj(prev_output_tokens)
        # below is the decoder operation
        bs, slen, dim = prev_output_tokens.size()
        alignment_layer = self.decoder.num_layers - 1

        enc: Optional[torch.Tensor] = None
        padding_mask: Optional[torch.Tensor] = None
        # embed positions
        positions = None
        if self.decoder.embed_positions is not None:
            positions = self.decoder.embed_positions(
                mask + self.decoder.embed_positions.padding_idx, incremental_state=None
            )

        # embed tokens and positions
        x = self.decoder.embed_scale * prev_output_tokens
        if self.decoder.project_in_dim is not None:
            x = self.decoder.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.decoder.layernorm_embedding is not None:
            x = self.decoder.layernorm_embedding(x)

        x = self.decoder.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[torch.Tensor] = None
        self_attn_padding_mask = ~(mask.bool())

        # decoder layers
        attn: Optional[torch.Tensor] = None
        inner_states: List[Optional[torch.Tensor]] = [x]
        for idx, layer in enumerate(self.decoder.layers):
            self_attn_mask = self.decoder.buffered_future_mask(x)

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state=None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.decoder.layer_norm is not None:
            x = self.decoder.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.decoder.project_out_dim is not None:
            x = self.decoder.project_out_dim(x)

        if self.output_layer_type == "simple_mlp":
            # return hidden_states instead for later loss computation
            logits = x
        elif self.output_layer_type == "linear":
            logits = self.output_proj(x)
        else:
            raise ValueError(f"output_layer {self.output_layer_type} not supported")

        if hasattr(self, "aux_output_proj"):
            aux_output = self.aux_output_proj(x)
        else:
            aux_output = None

        # remove the last frame
        return logits, aux_output