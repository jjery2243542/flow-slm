from pipeline import GSLMPipeline
import torch
from torch import nn
from losses import FlowLoss
import torch.nn.functional as F
from typing import Optional, Tuple

class Sampler(torch.nn.Module):
    def __init__(self, gslm_pipeline: GSLMPipeline, flow_loss: FlowLoss, frame_rate: int = 12.5, silence_indices: Optional[list] = None):
        super().__init__()
        self.gslm_pipeline = gslm_pipeline
        self.conf = self.gslm_pipeline.conf
        self.ssl_dim = self.conf.model.ssl_dim
        self.reduction_factor = self.conf.model.reduction_factor
        self.flow_loss = flow_loss
        self.frame_rate = frame_rate
        self.sigmoid = nn.Sigmoid()
        # default silence indices can be overridden via constructor
        self.silence_indices = silence_indices or [1049, 127, 1880, 1492, 972, 1031, 395, 2029, 581, 175, 1926, 407, 1316]

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Return logits with tokens beyond cumulative probability p filtered to -inf."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        batch_size = logits.size(0)
        unsorted_logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
        return unsorted_logits

    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Return logits with everything except top-k set to -inf."""
        top_k_vals, top_k_idx = torch.topk(logits, k, dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        batch_size = logits.size(0)
        filtered.scatter_(batch_size, top_k_idx, top_k_vals)
        return filtered

    def sample_from_logits(self, logits: torch.Tensor, topk: Optional[int] = None, topp: Optional[float] = None,
                           temperature: float = 1.0, penalize_silence: bool = False, penalize_weight: float = 10.0) -> torch.Tensor:
        """
        Sample indices from logits with optional top-k / top-p filtering, temperature scaling and silence penalization.
        Returns tensor shape [batch_size, 1].
        """
        if temperature != 1.0:
            logits = logits / temperature

        if penalize_silence and self.silence_indices:
            logits = logits.clone()
            logits[:, self.silence_indices] -= penalize_weight

        if topk is not None and topp is not None:
            raise ValueError("Cannot use both top-k and top-p sampling at the same time.")

        if topp is not None:
            filtered_logits = self._top_p_filter(logits, topp)
        elif topk is not None:
            filtered_logits = self._top_k_filter(logits, topk)
        else:
            filtered_logits = logits

        probs = F.softmax(filtered_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, num_samples=1)
        return sampled_indices

    def sample(self,
               batch_size: int = 1,
               min_len: float = 10,
               max_len: float = 15,
               threshold: float = 0.5,
               ode_steps: int = 64,
               device: str = "cuda",
               token_temperature: float = 1.0,
               temperature: float = 1.0,
               prompts: Optional[torch.Tensor] = None,
               solver: str = "euler",
               eos_aux_token: Optional[int] = None,
               cfg_scale: float = 0.3,
               topk: Optional[int] = None,
               topp: Optional[float] = 0.95,
               penalize_silence: bool = False,
               penalize_weight: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens using the GSLM pipeline and the flow model.
        Returns (generated_tokens_without_bos, stop_steps)
        """
        bos_token = torch.full((batch_size, 1), self.gslm_pipeline.bos_index, dtype=torch.long, device=device)
        bos_vec = self.gslm_pipeline.embed(bos_token)

        max_infer_steps = round(max_len * self.frame_rate / self.reduction_factor)

        prev_tokens = bos_vec if prompts is None else torch.cat([bos_vec, prompts.to(device)], dim=1)
        prev_tokens = prev_tokens.to(torch.bfloat16)

        has_ended = prev_tokens.new_zeros(batch_size, dtype=torch.bool)
        stop_steps = prev_tokens.new_zeros(batch_size, dtype=torch.int32)

        start_step = 0 if prompts is None else prompts.shape[1]

        for step in range(start_step, max_infer_steps):
            padding_mask = prev_tokens.new_ones((batch_size, prev_tokens.shape[1]))
            logits, aux_output = self.gslm_pipeline.decoder(prev_tokens, padding_mask)

            # handle extra_future_tokens -> split aux_output into chunks if configured
            if getattr(self.conf.model, "extra_future_tokens", 0) > 0:
                aux_output_chunk = torch.chunk(aux_output, self.conf.model.extra_future_tokens, dim=2)

            tokens = None
            # If an explicit aux eos token is provided and future_conditioning is off, sample from the next semantic token 
            if eos_aux_token is not None and not getattr(self.conf.model, "future_conditioning", False):
                tokens = self.sample_from_logits(aux_output_chunk[0][:, -1], topk=topk, topp=topp, temperature=token_temperature,
                                                 penalize_silence=penalize_silence, penalize_weight=penalize_weight)
                is_stop_token = tokens.squeeze(dim=1) == eos_aux_token
                end_at_this_step = is_stop_token & (stop_steps == 0)
            elif getattr(self.conf.model, 'future_conditioning', False):
                # sample tokens for each future head
                merge_tokens = []
                for k in range(self.conf.model.extra_future_tokens):
                    chunk_logits = aux_output_chunk[k][:, -1]
                    tok = self.sample_from_logits(chunk_logits, topk=topk, topp=topp, temperature=token_temperature,
                                                  penalize_silence=penalize_silence, penalize_weight=penalize_weight)
                    merge_tokens.append(tok)
                    if k == 0:
                        is_stop_token = tok.squeeze(dim=1) == eos_aux_token if eos_aux_token is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
                        end_at_this_step = is_stop_token & (stop_steps == 0)
                tokens = torch.stack(merge_tokens, dim=2)

            stop_steps[end_at_this_step] = step
            has_ended = has_ended | end_at_this_step
            if has_ended.all():
                break

            # Prepare z for the flow model
            if getattr(self.conf.model, "token_conditioning", False):
                tokens_without_eos = tokens.clone()
                # guard for missing special token embedding
                if not getattr(self.conf.model, "add_special_token_to_embedding_table", False):
                    tokens_without_eos[tokens >= 2048] = 2047
                token_embed = self.gslm_pipeline.token_embed(tokens_without_eos)
                if getattr(self.conf.model, "future_conditioning", False):
                    token_embed = torch.flatten(token_embed, start_dim=2, end_dim=-1)
                z = torch.cat([logits[:, -1:, :], token_embed], dim=2)
            else:
                z = logits[:, -1:, :]

            # Sample from the flow model (single call). capture norms if the flow returns them
            samples = self.flow_loss.sample(z, steps=ode_steps, temperature=temperature, solver=solver, cfg_scale=cfg_scale)
            prev_tokens = torch.cat([prev_tokens, samples.to(prev_tokens.dtype)], dim=1)

        if not has_ended.all():
            stop_steps[(stop_steps == 0)] = max_infer_steps

        return prev_tokens[:, 1:], stop_steps

