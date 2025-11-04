from typing import Optional, Tuple, Union, List
import math
import logging

import torch
import torch.nn as nn
import torchaudio
from contextlib import nullcontext

from transformers import MimiModel, AutoFeatureExtractor

from model_utils import modulate

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


class TimestepEmbedder(nn.Module):
	"""Embeds scalar timesteps into vector representations."""

	def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(frequency_embedding_size, hidden_size, bias=True),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size, bias=True),
		)
		self.frequency_embedding_size = frequency_embedding_size

	@staticmethod
	def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, scale: float = 1000.0) -> torch.Tensor:
		"""Create sinusoidal timestep embeddings.

		Args:
			t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
			dim: The dimension of the output.
			max_period: Controls the minimum frequency of the embeddings.
			scale: Scaling factor for the embeddings.

		Returns:
			An (N, D) Tensor of positional embeddings.
		"""
		# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
		half = dim // 2
		freqs = torch.exp(
			-math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype) / half
		).to(device=t.device)
		args = t[:, :, None].float() * scale * freqs[None]
		embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		if dim % 2:
			embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
		return embedding

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		"""Forward pass for timestep embedding.

		Args:
			t: Timestep tensor

		Returns:
			Embedded timestep tensor
		"""
		t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(t.dtype)
		t_emb = self.mlp(t_freq)
		return t_emb


class ResBlock(nn.Module):
	"""A residual block with adaptive layer normalization."""

	def __init__(self, channels: int):
		super().__init__()
		self.channels = channels

		self.in_ln = nn.LayerNorm(channels, eps=1e-6)
		self.mlp = nn.Sequential(
			nn.Linear(channels, channels, bias=True),
			nn.SiLU(),
			nn.Linear(channels, channels, bias=True),
		)

		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(channels, 3 * channels, bias=True)
		)

	def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		"""Forward pass for residual block with adaptive layer norm.

		Args:
			x: Input tensor
			y: Conditioning tensor

		Returns:
			Output tensor with residual connection
		"""
		shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
		h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
		h = self.mlp(h)
		return x + gate_mlp * h


class FinalLayer(nn.Module):
	"""The final layer adopted from DiT."""

	def __init__(self, model_channels: int, out_channels: int):
		super().__init__()
		self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
		self.linear = nn.Linear(model_channels, out_channels, bias=True)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(model_channels, 2 * model_channels, bias=True)
		)

	def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
		"""Forward pass for final layer.

		Args:
			x: Input tensor
			c: Conditioning tensor

		Returns:
			Output tensor
		"""
		shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
		x = modulate(self.norm_final(x), shift, scale)
		x = self.linear(x)
		return x


class SimpleMLPAdaLN(nn.Module):
	"""The MLP for Diffusion Loss with adaptive layer normalization."""

	def __init__(
		self,
		in_channels: int,
		model_channels: int,
		out_channels: int,
		z_channels: int,
		num_res_blocks: int,
		grad_checkpointing: bool = False
	):
		super().__init__()

		self.in_channels = in_channels
		self.model_channels = model_channels
		self.out_channels = out_channels
		self.num_res_blocks = num_res_blocks
		self.grad_checkpointing = grad_checkpointing

		self.time_embed = TimestepEmbedder(model_channels)
		self.cond_embed = nn.Linear(z_channels, model_channels)
		self.input_proj = nn.Linear(in_channels, model_channels)

		res_blocks = []
		for i in range(num_res_blocks):
			res_blocks.append(ResBlock(model_channels))

		self.res_blocks = nn.ModuleList(res_blocks)
		self.final_layer = FinalLayer(model_channels, out_channels)

		self.initialize_weights()

	def initialize_weights(self):
		"""Initialize model weights."""
		def _basic_init(module):
			if isinstance(module, nn.Linear):
				torch.nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
		self.apply(_basic_init)

		# Initialize timestep embedding MLP
		nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
		nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

		# Zero-out adaLN modulation layers
		for block in self.res_blocks:
			nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
			nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

		# Zero-out output layers
		nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
		nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
		nn.init.constant_(self.final_layer.linear.weight, 0)
		nn.init.constant_(self.final_layer.linear.bias, 0)

	def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
		"""Apply the model to an input batch.

		Args:
			x: An [N x T x C] Tensor of inputs
			t: An [N x T] timesteps
			c: Conditioning from AR transformer

		Returns:
			An [N x T x C] Tensor of outputs
		"""
		x = self.input_proj(x)
		t = self.time_embed(t)
		c = self.cond_embed(c)

		y = t + c

		if self.grad_checkpointing and not torch.jit.is_scripting():
			from torch.utils.checkpoint import checkpoint
			for block in self.res_blocks:
				x = checkpoint(block, x, y)
		else:
			for block in self.res_blocks:
				x = block(x, y)

		out = self.final_layer(x, y)
		return out


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
			if decoder_dim > 1280 and output_dim <= 1280:
				model_dim = decoder_dim 
			elif decoder_dim > 1280 and output_dim > 1280:
				model_dim = max(decoder_dim, output_dim)
			elif decoder_dim <= 1280 and output_dim < decoder_dim * 2:
				model_dim = decoder_dim * 2
			elif decoder_dim <= 1280 and output_dim >= decoder_dim * 2:
				model_dim = output_dim
			self.output_proj = SimpleMLPAdaLN(output_dim, model_dim, output_dim, decoder_dim + token_emb_dim, n_res_blocks)

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


class ELMDecoderWrapperWithText(BaseDecoderWrapper):
	"""Decoder wrapper that accepts an extra text input and emits an extra text output.

	The wrapper uses the provided `elm` model's input embeddings to embed the
	text input tokens, projects them to the decoder dimensionality, pools the
	text embeddings into a per-batch conditioning vector and adds that vector
	to the decoder token embeddings before running the transformer. It also
	exposes text logits produced by the ELM model's output projection (lm_head)
	computed from the final hidden states.
	"""

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
		token_emb_dim: int = 0,
		freeze_input_output_layer: bool = False, 
	):
		super().__init__(elm, input_dim, decoder_dim, output_dim, aux_output_dim, output_layer, n_res_blocks, aux_output_layer_idx, token_emb_dim)
		# Keep reference to the high-level ELM model so we can access its embeddings and output head
		self.elm = elm
		self.decoder = elm.transformer

		# Input token embedding module from the original ELM model
		# (expected to be an nn.Embedding)
		self.elm_token_embeddings = elm.get_input_embeddings()
		if self.elm.config.share_input_output_layers:
			self.text_output_proj = torch.nn.Linear(self.elm_token_embeddings.weight.size(1), self.elm_token_embeddings.weight.size(0), bias=False)
			self.text_output_proj.weight = self.elm_token_embeddings.weight
		else:
			self.text_output_proj = elm.lm_head

		if freeze_input_output_layer:
			for param in self.elm_token_embeddings.parameters():
				param.requires_grad = False
			for param in self.text_output_proj.parameters():
				param.requires_grad = False
		
	def forward(
		self,
		input_tokens: torch.Tensor = None,
		text_input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		cache_position: Optional[torch.Tensor] = None,
		text_attention_mask: Optional[torch.Tensor] = None,
		shift_audio_prediction: int = 2,
	):
		"""Forward pass that accepts an extra text input and returns text logits.

		Args:
			input_tokens: Main input tokens
			attention_mask: Attention mask for decoder
			past_key_values, use_cache, cache_position: standard caching params
			text_input_ids: Optional token ids for the extra text input (B x T_text)
			text_attention_mask: Optional mask for text input (B x T_text)
			shift_audio_prediction: Number of tokens to shift the audio prediction by

		Returns:
			(logits, aux_output, text_logits)
		"""
		# project main inputs
		inputs_embeds = self.input_proj(input_tokens)

		# compute text conditioning embeddings aligned with the main sequence and add
		if text_input_ids is not None:
			text_emb = self.elm_token_embeddings(text_input_ids)

			# masking the padding tokens from embeddings
			if text_attention_mask is not None:
				mask = text_attention_mask.to(text_emb.dtype).unsqueeze(-1)
				text_emb = text_emb * mask

			# shift the speech embeddings and text embeddings
			if shift_audio_prediction > 0:
				pad = torch.zeros(inputs_embeds.size(0), shift_audio_prediction, inputs_embeds.size(2), dtype=text_emb.dtype, device=text_emb.device)
				inputs_embeds = torch.cat([pad, inputs_embeds], dim=1)
			elif shift_audio_prediction < 0:
				pad = torch.zeros(text_emb.size(0), -shift_audio_prediction, text_emb.size(2), dtype=text_emb.dtype, device=text_emb.device)
				text_emb = torch.cat([pad, text_emb], dim=1)

			target_len = inputs_embeds.shape[1]
			text_len = text_emb.shape[1]

			if text_len < target_len:
				pad_len = target_len - text_len
				pad = torch.zeros(text_emb.size(0), pad_len, text_emb.size(2), dtype=text_emb.dtype, device=text_emb.device)
				text_emb = torch.cat([text_emb, pad], dim=1)
			elif text_len > target_len:
				pad_len = text_len - target_len
				pad = torch.zeros(inputs_embeds.size(0), pad_len, inputs_embeds.size(2), dtype=text_emb.dtype, device=text_emb.device)
				inputs_embeds = torch.cat([inputs_embeds, pad], dim=1)
			else:
				raise ValueError("text_input_ids length must be less than or equal to input_tokens length")
			
			inputs_embeds = inputs_embeds + text_emb

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
			logits = hidden_states
		elif self.output_layer_type == "linear":
			logits = self.output_proj(hidden_states)
		else:
			raise ValueError(f"output_layer {self.output_layer_type} not supported")

		if hasattr(self, "aux_output_proj"):
			aux_output = self.aux_output_proj(aux_hidden_states)
		else:
			aux_output = None
		text_logits = self.text_output_proj(hidden_states)
		text_length = text_attention_mask.size(1) 

		if shift_audio_prediction >= 0:
			text_logits = text_logits[:, :text_length, :]
		else:
			text_logits = text_logits[:, -shift_audio_prediction:text_length - shift_audio_prediction, :]

		return logits, aux_output, text_logits
