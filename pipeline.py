import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import length_to_mask
from model import MimiEncoder, ELMDecoderWrapper, ELMDecoderWrapperWithText
from model_utils import reduce_features, split_features
from transformers import AutoModelForCausalLM


class GSLMPipeline(nn.Module):
    def __init__(self, conf, args):
        super().__init__()
        self.conf = conf
        self.args = args
        
        if hasattr(self.conf.model, "ssl_model") and self.conf.model.ssl_model == "mimi":
            n_quantizers = getattr(self.conf.model, "n_quantizers", 0)
            self.ssl_model = MimiEncoder(freeze=self.conf.model.freeze, n_quantizers=n_quantizers)

        # Initialize decoder model
        if "OpenELM" in self.conf.model.decoder:
            model_name = f"apple/{self.conf.model.decoder}"
        else:
            raise NotImplementedError(f"Decoder model {self.conf.model.decoder} not supported.")

        attn_implementation = "flash_attention_2" if self.conf.model.flash_attention else "eager"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or attn_implementation == 'flash_attention_2' else torch.float32
        decoder_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        # Initialize normalization (moved to helper)
        self._init_normalization()
        # Initialize remaining model components (moved to helper)
        self._init_model_components(decoder_model)

        # Initialize embeddings (moved to helper)
        self._init_embeddings()

    @property
    def _decoder_model(self):
        return self.decoder.lm

    def _init_normalization(self):
        """Load and register static normalization buffers if configured.

        This was previously in __init__; extracted for clarity and reuse.
        """
        if hasattr(self.conf.model, "norm") and self.conf.model.norm == "static":
            mean = np.load(self.conf.model.mean_path)
            self.register_buffer('mean', torch.from_numpy(mean).float())
            std = np.load(self.conf.model.std_path)
            self.register_buffer('std', torch.from_numpy(std).float())
    
    def _init_model_components(self, decoder_model):
        """Initialize dims, aux outputs, decoder wrapper and related model components."""
        ssl_dim, reduction_factor = self.conf.model.ssl_dim, self.conf.model.reduction_factor
        if self.conf.optimizer.loss_function == "FM":
            self.input_dim = ssl_dim * reduction_factor
            self.output_dim = ssl_dim * reduction_factor
        else:
            raise NotImplementedError(f"Loss function {self.conf.optimizer.loss_function} not supported.")

        if (self.conf.model.extra_future_tokens > 1 or self.conf.model.future_conditioning) and self.conf.model.reduction_factor > 1:
            raise ValueError("extra_future_tokens > 1 is not supported when reduction_factor > 1.")

        # Initialize auxiliary output dimensions for token prediction
        if self.conf.model.ssl_model == "mimi" and self.conf.optimizer.token_loss_weight > 0:
            n_special_tokens = getattr(self.conf.model, "n_special_tokens", 0)
            self.aux_output_dim = self.ssl_model.model.config.codebook_size + n_special_tokens
            # hardcoded: use the last index as eos token
            self.eos_token_index = self.aux_output_dim - 1
            if hasattr(self.conf.model, "extra_future_tokens") and self.conf.model.extra_future_tokens > 0:
                self.aux_output_dim = self.aux_output_dim * (self.conf.model.extra_future_tokens * reduction_factor)
        else:
            self.aux_output_dim = None

        # Initialize token embedding dimensions
        self.token_emb_dim = self.conf.model.token_emb_dim if hasattr(self.conf.model, "token_emb_dim") and hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning else 0
        if hasattr(self.conf.model, "future_conditioning") and self.conf.model.future_conditioning:
            self.token_emb_dim *= self.conf.model.extra_future_tokens
        self.token_emb_dim = self.token_emb_dim * reduction_factor

        if "OpenELM" in self.conf.model.decoder:
            output_layer = "simple_mlp" if self.conf.optimizer.loss_function == "FM" else "linear"
            self._output_layer_type = output_layer
            self._n_res_blocks = self.conf.model.n_res_blocks
            self.aux_output_layer_idx = None if not hasattr(self.conf.model, "aux_output_layer_idx") else self.conf.model.aux_output_layer_idx

            if hasattr(self.conf.model, "ssl_model") and self.conf.model.ssl_model == "mimi":
                self.decoder = ELMDecoderWrapper(
                    decoder_model,
                    input_dim=self.input_dim,
                    decoder_dim=self.conf.model.decoder_dim,
                    output_dim=self.output_dim,
                    aux_output_dim=self.aux_output_dim,
                    output_layer=self._output_layer_type,
                    n_res_blocks=self._n_res_blocks,
                    aux_output_layer_idx=self.aux_output_layer_idx,
                    token_emb_dim=self.token_emb_dim,
                )
            # use self._lm for config access
            self.pad_index = decoder_model.config.pad_token_id
            self.bos_index = decoder_model.config.bos_token_id
            self.eos_index = decoder_model.config.eos_token_id

    def _init_embeddings(self):
        """Create embedding layers (input BOS embedding and optional token embeddings)."""
        # Initialize embeddings for bos
        self.embed = nn.Embedding(3, embedding_dim=self.input_dim)

        nn.init.normal_(self.embed.weight, mean=0, std=self.input_dim ** -0.5)
        #nn.init.constant_(self.embed.weight[self.pad_index], 0)

        # Initialize token embeddings if needed
        if hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning:
            # add token emb to z, only support mimi
            if hasattr(self.conf.model, "add_special_token_to_embedding_table") and self.conf.model.add_special_token_to_embedding_table:
                self.token_embed = nn.Embedding(self.ssl_model.model.config.codebook_size + self.conf.model.n_special_tokens, embedding_dim=self.conf.model.token_emb_dim)
            else:
                self.token_embed = nn.Embedding(self.ssl_model.model.config.codebook_size, embedding_dim=self.conf.model.token_emb_dim)
    
    def _decode(self, prev_tokens, wav_len, padding_mask=None, **decoder_kwargs):
        """Decode helper. If padding_mask is provided it will be used, otherwise it's computed from wav_len.

        Returns (logits, aux_output).
        """
        if padding_mask is None:
            ssl_feats_len = prev_tokens.shape[1] - 1
            abs_len = torch.round(wav_len * ssl_feats_len).long() + 1
            padding_mask = length_to_mask(abs_len, max_len=prev_tokens.shape[1], dtype=torch.bool)
        logits, aux_output = self.decoder(prev_tokens, padding_mask)
        return logits, aux_output

    def _get_ssl_feats(self, wavs, wav_len):
        with torch.no_grad():
            if self.conf.model.ssl_model == "mimi" and hasattr(self.conf.model, "n_quantizers") and self.conf.model.n_quantizers > 0:
                ssl_feats, tokens = self.ssl_model(wavs, wav_len)
            else:
                raise NotImplementedError(f"SSL model {self.conf.model.ssl_model} not supported.")

            ssl_abs_len = torch.round(wav_len * ssl_feats.shape[1]).long()
            #ssl_padding_mask = ~length_to_mask(ssl_abs_len, dtype=torch.bool)
            
            if hasattr(self.conf.model, "norm") and self.conf.model.norm == "static":
                ssl_feats = (ssl_feats - self.mean) / self.std

            # Reduce features
            if self.conf.model.reduction_factor > 1:
                reduced_ssl_feats = reduce_features(ssl_feats, self.conf.model.reduction_factor, pad=False)
            else:
                reduced_ssl_feats = ssl_feats
        return reduced_ssl_feats, ssl_feats, ssl_abs_len, tokens

    def _process_token_predictions(self, aux_output, wav_len, tokens, bs):
        """Extracted logic for processing token predictions.

        Returns (token_logits, tokens, split_padding_mask).
        If tokens is None or token prediction is disabled, returns (None, None, None).
        """
        # If tokens is None, nothing to do
        if tokens is None:
            return None, None, None

        if self.conf.model.ssl_model == "mimi" and (
            self.conf.optimizer.token_loss_weight > 0 or self.conf.model.token_conditioning
        ) and self.conf.model.n_quantizers > 0:
            token_logits = split_features(aux_output, self.conf.model.reduction_factor)  # [B, T * r, F // r]
            k = 1 if not hasattr(self.conf.model, "extra_future_tokens") or self.conf.model.extra_future_tokens == 0 else self.conf.model.extra_future_tokens
            ssl_abs_len = torch.round(wav_len * tokens.shape[1]).long()
            # add one for eos token
            split_padding_mask = length_to_mask(ssl_abs_len + 1, dtype=torch.bool)
            # append eos as last k tokens 
            eos_index = self.eos_token_index
            tokens = torch.cat([tokens, tokens.new_ones((bs, k, 1)).long() * eos_index], dim=1)  # shape [B, T + k, 1]
            offsets = ssl_abs_len.unsqueeze(1) + torch.arange(k, device=tokens.device).unsqueeze(0)  # shape [B, k]
            batch_indices = torch.arange(bs, device=tokens.device).unsqueeze(1).expand(bs, k)         # shape [B, k]
            tokens[batch_indices, offsets] = eos_index
            token_logits = token_logits[:, :tokens.shape[1], :]  # align with tokens
        else:
            token_logits = None
            tokens = None
            split_padding_mask = None

        return token_logits, tokens, split_padding_mask

    def _apply_token_conditioning_and_padding(self, logits, tokens, padding_mask, abs_len, bs):
        """Apply token conditioning to logits (if enabled) and compute padding_mask_for_loss.

        Returns (logits, padding_mask_for_loss).
        """
        # Apply token conditioning if specified
        if hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning:
            L = logits.shape[1]
            if hasattr(self.conf.model, "future_conditioning") and self.conf.model.future_conditioning:
                k = 1 if self.conf.model.extra_future_tokens == 0 else self.conf.model.extra_future_tokens
                conditioning_tokens = torch.stack([tokens[:, kk:kk+L, 0] for kk in range(k)], dim=2) # [B, T, k]
                token_embed = self.token_embed(conditioning_tokens).flatten(start_dim=2, end_dim=-1) # [B, T, k * D]
                logits = torch.cat([logits, token_embed], dim=2)
            elif self.conf.model.reduction_factor > 1:
                token_embed = self.token_embed(tokens[:, :L * self.conf.model.reduction_factor, 0])
                token_embed = reduce_features(token_embed, self.conf.model.reduction_factor, pad=False)
                logits = torch.cat([logits, token_embed], dim=2)
            elif self.conf.model.reduction_factor == 1:
                # use only the first token
                token_embed = self.token_embed(tokens[:, :L, 0])
                logits = torch.cat([logits, token_embed], dim=2)

        # Remove one frame for loss computation
        padding_mask_for_loss = padding_mask.clone()
        padding_mask_for_loss[torch.arange(bs, device=padding_mask_for_loss.device), abs_len - 1] = 0
        padding_mask_for_loss = padding_mask_for_loss[:, :-1].unsqueeze(dim=2)

        return logits, padding_mask_for_loss

    def forward(self, wavs, wav_len, **decoder_kwargs):
        reduced_ssl_feats, ssl_feats, ssl_abs_len, tokens = self._get_ssl_feats(wavs, wav_len)

        # Prepending BOS token
        bs = reduced_ssl_feats.shape[0]
        bos_token = ssl_feats.new_ones((bs, 1)).long() * self.bos_index
        bos_vec = self.embed(bos_token)

        prev_tokens = torch.cat([bos_vec, reduced_ssl_feats], dim=1)
        # compute lengths/padding so they're available for loss computation later
        ssl_feats_len = prev_tokens.shape[1] - 1
        abs_len = torch.round(wav_len * ssl_feats_len).long() + 1
        padding_mask = length_to_mask(abs_len, max_len=prev_tokens.shape[1], dtype=torch.bool)

        logits, aux_output = self._decode(prev_tokens, wav_len, padding_mask=padding_mask, **decoder_kwargs)

        # Remove the last frame because there is no loss applied on it (eos token)
        logits = logits[:, :-1]

        # Process token predictions (delegated to helper)
        token_logits, tokens, split_padding_mask = self._process_token_predictions(aux_output, wav_len, tokens, bs)

        # Apply token conditioning if specified and compute padding mask for loss
        logits, padding_mask_for_loss = self._apply_token_conditioning_and_padding(logits, tokens, padding_mask, abs_len, bs)

        return logits, reduced_ssl_feats, padding_mask_for_loss, token_logits, tokens, split_padding_mask


class GSLMWithTextPipeline(GSLMPipeline):
    def __init__(self, conf, args):
        super().__init__(conf, args)
        freeze_text_io = getattr(self.conf.model, "freeze_text_input_output", True)
        self.decoder = ELMDecoderWrapperWithText(
            self._decoder_model,
            input_dim=self.input_dim,
            decoder_dim=self.conf.model.decoder_dim,
            output_dim=self.output_dim,
            aux_output_dim=self.aux_output_dim,
            output_layer=self._output_layer_type,
            n_res_blocks=self._n_res_blocks,
            aux_output_layer_idx=self.aux_output_layer_idx,
            token_emb_dim=self.token_emb_dim,
            freeze_input_output_layer=freeze_text_io,
        )

    def _decode(self, prev_tokens, wav_len, padding_mask=None, text_input_ids=None, text_attention_mask=None, shift_audio_prediction=2, **decoder_kwargs):
        """Decode that accepts text input and records text_logits for later retrieval.

        Returns (logits, aux_output) to match base class contract but stores text logits
        in self._decoder_last_extras["text_logits"].
        """
        if padding_mask is None:
            ssl_feats_len = prev_tokens.shape[1] - 1
            abs_len = torch.round(wav_len * ssl_feats_len).long() + 1
            padding_mask = length_to_mask(abs_len, max_len=prev_tokens.shape[1], dtype=torch.bool)

        # ELMDecoderWrapperWithText expects attention_mask and text inputs as kwargs
        logits, aux_output, text_logits = self.decoder(
            prev_tokens,
            text_input_ids=text_input_ids,
            attention_mask=padding_mask,
            text_attention_mask=text_attention_mask,
            shift_audio_prediction=shift_audio_prediction,
            **decoder_kwargs,
        )
        return logits, aux_output, text_logits

    def forward(self, wavs, wav_len, text_input_ids, text_attention_mask=None, shift_audio_prediction=2, **decoder_kwargs):
        # Re-implement the original GSLMPipeline.forward steps here so we don't call super().
        reduced_ssl_feats, ssl_feats, ssl_abs_len, tokens = self._get_ssl_feats(wavs, wav_len)

        # Prepending BOS token
        bs = reduced_ssl_feats.shape[0]
        bos_token = ssl_feats.new_ones((bs, 1)).long() * self.bos_index
        # use different bos for text and audio to avoid confusion
        bos_vec = self.embed(bos_token)
        prev_tokens = torch.cat([bos_vec, reduced_ssl_feats], dim=1)
        # compute lengths/padding so they're available for loss computation later
        ssl_feats_len = prev_tokens.shape[1] - 1
        abs_len = torch.round(wav_len * ssl_feats_len).long() + 1
        padding_mask = length_to_mask(abs_len, max_len=prev_tokens.shape[1], dtype=torch.bool)

        # prepend BOS token for text 
        if text_input_ids is not None:
            text_bos_token = text_input_ids.new_ones((bs, 1)) * self.bos_index
            text_input_ids_w_bos = torch.cat([text_bos_token, text_input_ids], dim=1)
            text_eos_token = text_input_ids.new_ones((bs, 1)) * self.eos_index
            text_input_ids_w_eos = torch.cat([text_input_ids, text_eos_token], dim=1)

        if text_attention_mask is not None:
            text_bos_mask = text_attention_mask.new_ones((bs, 1))
            text_attention_mask_w_bos = torch.cat([text_bos_mask, text_attention_mask], dim=1)

        
        # Decode with text inputs forwarded to the text-aware decoder
        logits, aux_output, text_logits = self._decode(
            prev_tokens,
            wav_len,
            padding_mask=padding_mask,
            text_input_ids=text_input_ids_w_bos,
            text_attention_mask=text_attention_mask_w_bos if text_attention_mask is not None else None,
            **decoder_kwargs,
        )

        start = shift_audio_prediction if shift_audio_prediction >= 0 else 0
        length = reduced_ssl_feats.shape[1]
        # Remove the last frame because there is no loss applied on it (eos token)
        logits = logits[:, start:start + length]
        aux_output = aux_output[:, start:start + length + 1]

        # Process token predictions (delegated to helper)
        token_logits, tokens, split_padding_mask = self._process_token_predictions(aux_output, wav_len, tokens, bs)

        # Apply token conditioning if specified and compute padding mask for loss
        logits, padding_mask_for_loss = self._apply_token_conditioning_and_padding(logits, tokens, padding_mask, abs_len, bs)
        return logits, reduced_ssl_feats, padding_mask_for_loss, token_logits, tokens, split_padding_mask, text_logits, text_input_ids_w_eos, text_attention_mask_w_bos
