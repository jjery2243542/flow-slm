from pipeline import GSLMPipeline
import torch
from torch import nn
from losses import FlowLoss
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)

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
               ode_steps: int = 64,
               device: str = "cuda",
               text_temperature: float = 1.0,
               token_temperature: float = 1.0,
               temperature: float = 1.0,
               use_text_prompt: bool = False,
               shift_audio_prediction: int = 0,
               audio_prompts: Optional[torch.Tensor] = None,
               text_prompts: Optional[torch.Tensor] = None,
               text_attention_mask: Optional[torch.Tensor] = None,
               solver: str = "euler",
               eos_aux_token: Optional[int] = None,
               cfg_scale: float = 0.3,
               schedule: str = "linear",
               shift_alpha: float = 1.0,
               topk: Optional[int] = None,
               topp: Optional[float] = 0.95,
               penalize_silence: bool = False,
               penalize_weight: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens using the GSLM pipeline and the flow model.
        Returns (generated_tokens_without_bos, stop_steps)
        """
        if eos_aux_token is None:
            raise ValueError("eos_aux_token must be provided for stopping criteria.")

        bos_token = torch.full((batch_size, 1), self.gslm_pipeline.bos_index, dtype=torch.long, device=device)
        bos_vec = self.gslm_pipeline.embed(bos_token)

        max_infer_steps = round(max_len * self.frame_rate / self.reduction_factor)

        prev_tokens = bos_vec if audio_prompts is None else torch.cat([bos_vec, audio_prompts.to(device)], dim=1)
        prev_tokens = prev_tokens.to(torch.bfloat16)

        if use_text_prompt:
            if text_prompts is None or text_attention_mask is None:
                raise ValueError("text_prompts and text_attention_mask must be provided when use_text_prompt is True.")
            text_input_ids = text_prompts.to(device).long()
            text_attention_mask = text_attention_mask.to(device).clone().long()
            text_has_ended = text_input_ids.new_zeros(batch_size, dtype=torch.bool, device=device)
            text_eos_index = self.gslm_pipeline.eos_index
            audio_prompt_length = prev_tokens.shape[1] if audio_prompts is not None else 0
            text_length = text_input_ids.shape[1]
            diff = audio_prompt_length + shift_audio_prediction - text_length
            print("diff", diff)
            print("before: text_input_ids.shape", text_input_ids.shape, "prev_tokens.shape", prev_tokens.shape)

            # sample text tokens until the audio_prompts length
            if diff > 0:
                for diff_step in range(diff):
                    outputs = self.gslm_pipeline.decoder(
                        input_tokens=prev_tokens[:, : text_length + diff_step], 
                        attention_mask=prev_tokens.new_ones((batch_size, text_length + diff_step)), 
                        text_input_ids=text_input_ids, 
                        text_attention_mask=text_attention_mask,
                        shift_audio_prediction=shift_audio_prediction,
                    )
                    logits, aux_output, text_logits = outputs
                    next_text_logits = text_logits[:, -1, :]
                    sampled_text = self.sample_from_logits(next_text_logits, topk=topk, topp=topp, temperature=text_temperature)
                    text_input_ids = torch.cat([text_input_ids, sampled_text.to(text_input_ids.dtype)], dim=1)
                    is_text_eos = sampled_text.squeeze(dim=1) == text_eos_index
                    text_has_ended = text_has_ended | is_text_eos
                    print(diff_step, text_attention_mask, text_input_ids.shape, text_attention_mask.shape, text_has_ended, tokenizer.batch_decode(text_input_ids))

                    if text_attention_mask.shape[1] < text_input_ids.shape[1]:
                        to_append = text_attention_mask.new_ones((batch_size, text_input_ids.shape[1] - text_attention_mask.shape[1]))
                        to_append = to_append * (~text_has_ended).long().unsqueeze(1)
                        text_attention_mask = torch.cat([text_attention_mask, to_append], dim=1)

            print("after: text_input_ids.shape", text_input_ids.shape)

        print("text_input_ids.shape", text_input_ids.shape)
        print("prev_tokens.shape", prev_tokens.shape)
        print("text", tokenizer.batch_decode(text_input_ids))
        print("text_attention_mask", text_attention_mask)
        print("finished text", text_has_ended)
        has_ended = prev_tokens.new_zeros(batch_size, dtype=torch.bool)
        stop_steps = prev_tokens.new_zeros(batch_size, dtype=torch.int32)

        start_step = 0 if audio_prompts is None else audio_prompts.shape[1]

        for step in range(start_step, max_infer_steps):

            padding_mask = prev_tokens.new_ones((batch_size, prev_tokens.shape[1]))
 
            outputs = self.gslm_pipeline.decoder(
                input_tokens=prev_tokens, 
                attention_mask=padding_mask, 
                text_input_ids=text_input_ids[:, :prev_tokens.shape[1]] if use_text_prompt else None, 
                text_attention_mask=text_attention_mask[:, :prev_tokens.shape[1]] if use_text_prompt else None,
                shift_audio_prediction=shift_audio_prediction,
            )
            if use_text_prompt and text_input_ids.shape[1] == prev_tokens.shape[1] + shift_audio_prediction:
                logits, aux_output, text_logits = outputs
                next_text_logits = text_logits[:, -1, :]
                sampled_text = self.sample_from_logits(next_text_logits, topk=topk, topp=topp, temperature=text_temperature)
                text_input_ids = torch.cat([text_input_ids, sampled_text.to(text_input_ids.dtype)], dim=1)
                is_text_eos = sampled_text.squeeze(dim=1) == text_eos_index
                text_has_ended = text_has_ended | is_text_eos

                if text_attention_mask.shape[1] < text_input_ids.shape[1]:
                    to_append = text_attention_mask.new_ones((batch_size, text_input_ids.shape[1] - text_attention_mask.shape[1]))
                    to_append = to_append * (~text_has_ended).long().unsqueeze(1)
                    text_attention_mask = torch.cat([text_attention_mask, to_append], dim=1)

            elif use_text_prompt and text_input_ids.shape[1] > prev_tokens.shape[1] + shift_audio_prediction:
                print("Warning: text_input_ids length greater than prev_tokens length. This should have been handled above.")
            elif use_text_prompt and text_input_ids.shape[1] < prev_tokens.shape[1] + shift_audio_prediction:
                raise ValueError("text_input_ids length cannot be less than prev_tokens length. It should already be handled above.")
            else:
                logits, aux_output = outputs

            # handle extra_future_tokens -> split aux_output into chunks if configured
            if getattr(self.conf.model, "extra_future_tokens", 1) > 1 and self.conf.model.reduction_factor > 1:
                raise ValueError("extra_future_tokens > 1 is not supported when reduction_factor > 1.")

            split_size = self.conf.model.extra_future_tokens * self.conf.model.reduction_factor
            aux_output_chunk = torch.chunk(aux_output, split_size, dim=2)

            tokens = None
            if not getattr(self.conf.model, "future_conditioning", False) and self.conf.model.reduction_factor == 1:
                tokens = self.sample_from_logits(aux_output_chunk[0][:, -1], topk=topk, topp=topp, temperature=token_temperature,
                                                 penalize_silence=penalize_silence, penalize_weight=penalize_weight)
                is_stop_token = tokens.squeeze(dim=1) == eos_aux_token
                end_at_this_step = is_stop_token & (stop_steps == 0)
            elif getattr(self.conf.model, 'future_conditioning', False) or self.conf.model.reduction_factor > 1:
                # sample tokens for each future head
                merge_tokens = []
                for k in range(split_size):
                    chunk_logits = aux_output_chunk[k][:, -1]
                    tok = self.sample_from_logits(chunk_logits, topk=topk, topp=topp, temperature=token_temperature,
                                                  penalize_silence=penalize_silence, penalize_weight=penalize_weight)
                    merge_tokens.append(tok)
                if getattr(self.conf.model, "extra_future_tokens", 1) > 1:
                    tok = merge_tokens[0]
                    is_stop_token = tok.squeeze(dim=1) == eos_aux_token
                    end_at_this_step = is_stop_token & (stop_steps == 0)
                elif self.conf.model.reduction_factor > 1:
                    is_stop_token = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    for tok in merge_tokens:
                        is_stop_token |= tok.squeeze(dim=1) == eos_aux_token
                    end_at_this_step = is_stop_token & (stop_steps == 0)
                tokens = torch.stack(merge_tokens, dim=2)

            stop_steps[end_at_this_step] = step
            has_ended = has_ended | end_at_this_step
            if has_ended.all():
                break

            # Prepare z for the flow model
            if getattr(self.conf.model, "token_conditioning", False):
                #tokens_without_eos = tokens.clone()
                ## guard for missing special token embedding
                #if not getattr(self.conf.model, "add_special_token_to_embedding_table", False):
                #    tokens_without_eos[tokens >= 2048] = 2047
                token_embed = self.gslm_pipeline.token_embed(tokens)
                if getattr(self.conf.model, "future_conditioning", False) or self.conf.model.reduction_factor > 1:
                    token_embed = torch.flatten(token_embed, start_dim=2, end_dim=-1)
                z = torch.cat([logits[:, -1:, :], token_embed], dim=2)
            else:
                z = logits[:, -1:, :]

            # Sample from the flow model (single call). capture norms if the flow returns them
            samples = self.flow_loss.sample(z, steps=ode_steps, temperature=temperature, solver=solver, cfg_scale=cfg_scale, schedule=schedule, shift_alpha=shift_alpha)
            prev_tokens = torch.cat([prev_tokens, samples.to(prev_tokens.dtype)], dim=1)

        if not has_ended.all():
            stop_steps[(stop_steps == 0)] = max_infer_steps

        if use_text_prompt:
            return prev_tokens[:, 1:], stop_steps, text_input_ids, text_attention_mask
        else:
            return prev_tokens[:, 1:], stop_steps

