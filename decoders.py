import torch
from typing import Optional, List
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
