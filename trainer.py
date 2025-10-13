"""Training module for continuous GSLM.

This module contains the main training class LanguageModeling which handles
the training, validation, and prediction steps for the continuous GSLM model.
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics.functional.classification import binary_f1_score
from speechbrain.dataio.dataio import length_to_mask
import numpy as np

from pipeline import GSLMPipeline
from losses import FlowLoss
from decode import compute_log_likelihood

class LanguageModeling(pl.LightningModule): 
    """Main training class for continuous GSLM.
    
    This class handles the training, validation, and prediction steps
    for the continuous GSLM model using PyTorch Lightning.
    """
    
    def __init__(self, args, conf): 
        super().__init__()
        self.count = 0
        self.args = args
        self.conf = conf
        conf_dict = self.conf.toDict()
        self.save_hyperparameters(conf_dict)

        self.gslm_pipeline = GSLMPipeline(conf, args)
        if hasattr(self.conf.training, "freeze_decoder") and self.conf.training.freeze_decoder:
            print("freeze decoder")
            self.gslm_pipeline.decoder.freeze_decoder()

        if self.conf.optimizer.loss_function == "MSE":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        elif self.conf.optimizer.loss_function == "L1":
            self.loss_fn = torch.nn.L1Loss(reduction="none") 
        elif self.conf.optimizer.loss_function == "GMM":
            self.loss_fn = GMMLoss(k=self.conf.model.k_mixtures, d=self.conf.model.ssl_dim * self.conf.model.reduction_factor)
        elif self.conf.optimizer.loss_function == "FM":
            dim = self.conf.model.ssl_dim if not hasattr(self.conf.model, "pca_module_path") else self.gslm_pipeline.pca_module.output_dim
            if hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning:
                if not hasattr(self.conf.model, "future_conditioning") or not self.conf.model.future_conditioning:
                    z_dim = self.conf.model.decoder_dim + self.conf.model.token_emb_dim
                else:
                    z_dim = self.conf.model.decoder_dim + self.conf.model.token_emb_dim * self.conf.model.extra_future_tokens
            else:
                z_dim = self.conf.model.decoder_dim
            null_prob = 0.0 if not hasattr(self.conf.optimizer, "null_prob") else self.conf.optimizer.null_prob
            self.loss_fn = FlowLoss(target_dim=dim * self.conf.model.reduction_factor, z_dim=z_dim, net=self.gslm_pipeline.decoder.output_proj, sigma_min=self.conf.optimizer.sigma_min, t_dist=self.conf.optimizer.t_dist, null_prob=null_prob)
        else:
            raise NotImplementedError(f"{self.conf.optimizer.loss_function} not implemented.")
        self.stop_token_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.Tensor([self.conf.optimizer.pos_weight]))
        self.token_loss_fn = torch.nn.CrossEntropyLoss(reduction="none") 

        if not hasattr(self.conf.optimizer, "loss_weight"):
            self.conf.optimizer.loss_weight = 1.0

    def configure_optimizers(self): 
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        if self.conf.optimizer.name == "AdamW":
            opt = torch.optim.AdamW(trainable_params, lr=self.conf.optimizer.lr, betas=self.conf.optimizer.betas, weight_decay=self.conf.optimizer.weight_decay, eps=self.conf.optimizer.eps)
        elif self.conf.optimizer.name == "AdamW8bit":
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(trainable_params, lr=self.conf.optimizer.lr, betas=self.conf.optimizer.betas, weight_decay=self.conf.optimizer.weight_decay, eps=self.conf.optimizer.eps, percentile_clipping=self.conf.optimizer.percentile_clipping)
        else:
            raise NotImplementedError(f"{self.conf.optimizer.name} not implemented.")

        from ..scheduler import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            opt, 
            num_warmup_steps=self.conf.training.num_warmup_steps, 
            num_training_steps=self.conf.training.max_steps, 
            min_lr_ratio=self.conf.training.min_lr_ratio
        )

        lr_scheduler_config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
        return lr_scheduler_config
    
    def forward(self, batch, reduction='token', likelihood=False):
        ids, wavs, wav_len = batch
        # make sure the mask is correctly computed
        wav_len = wav_len.float()

        if self.trainer.global_step < self.conf.training.frozen_decoder_steps and not self.gslm_pipeline.decoder.frozen:
            self.gslm_pipeline.decoder.freeze_decoder()
        elif self.trainer.global_step >= self.conf.training.frozen_decoder_steps and self.gslm_pipeline.decoder.frozen:
            self.gslm_pipeline.decoder.unfreeze_decoder()

        if self.training:
            if self.gslm_pipeline.decoder.frozen:
                with torch.no_grad():
                    logits, ssl_feats, padding_mask, is_stop_token, stop_token_target, stop_token_padding_mask, token_logits, tokens, token_padding_mask = self.gslm_pipeline(wavs, wav_len)
            else:
                logits, ssl_feats, padding_mask, is_stop_token, stop_token_target, stop_token_padding_mask, token_logits, tokens, token_padding_mask = self.gslm_pipeline(wavs, wav_len)

            if self.conf.optimizer.loss_weight > 0:
                loss = self.loss_fn(logits, ssl_feats)
            else:
                loss = torch.zeros_like(logits)
            if hasattr(self.conf.optimizer, "stop_token_weight") and self.conf.optimizer.stop_token_weight > 0:
                stop_token_loss = self.stop_token_loss_fn(is_stop_token, stop_token_target)
            else:
                stop_token_loss = torch.zeros_like(is_stop_token)
        else:
            with torch.no_grad():
                self.gslm_pipeline.eval()
                logits, ssl_feats, padding_mask, is_stop_token, stop_token_target, stop_token_padding_mask, token_logits, tokens, token_padding_mask = self.gslm_pipeline(wavs, wav_len)
                loss = self.loss_fn(logits, ssl_feats)
                stop_token_loss = self.stop_token_loss_fn(is_stop_token, stop_token_target)

        if self.conf.optimizer.token_loss_weight > 0:
            if hasattr(self.conf.model, "extra_future_tokens") and self.conf.model.extra_future_tokens > 0:
                token_losses = token_padding_mask.new_zeros((token_padding_mask.shape[0], token_padding_mask.shape[1])).float()
                token_weight = token_padding_mask.new_zeros((token_padding_mask.shape[0], token_padding_mask.shape[1])).float()

                token_logits_i = torch.chunk(token_logits, self.conf.model.extra_future_tokens, dim=2)
                k_future_tokens = self.conf.model.extra_future_tokens if self.training else self.args.use_k_future_tokens
                L = token_padding_mask.shape[1]
                for i in range(k_future_tokens):
                    logits_i = token_logits_i[i].reshape(-1, token_logits_i[i].shape[-1])
                    tokens_i = tokens[:, i: i + L].reshape(-1)
                    token_loss = self.token_loss_fn(logits_i, tokens_i)
                    token_loss = token_loss.reshape(token_logits_i[i].shape[0], token_logits_i[i].shape[1])
                    if self.args.ignore_eos and not self.training:
                        token_padding_mask_no_eos  = (token_padding_mask * (tokens_i.reshape(token_logits_i[i].shape[0], token_logits_i[i].shape[1]) != self.gslm_pipeline.eos_token_index)).float()
                        token_losses += token_loss * token_padding_mask_no_eos
                        token_weight += token_padding_mask_no_eos
                    else:
                        token_losses += token_loss * token_padding_mask
                        token_weight += token_padding_mask
                # only divide by the number of tokens that are not masked
                token_weight[token_weight == 0] = 1e-6
                token_loss = token_losses / token_weight
            else:
                token_loss = self.token_loss_fn(token_logits.reshape(-1, token_logits.shape[-1]), tokens.reshape(-1)).reshape(token_logits.shape[0], token_logits.shape[1])
        else:
            token_loss = None

        if reduction == "token":
            loss = torch.sum(loss * padding_mask) / (torch.sum(padding_mask) * self.conf.model.reduction_factor * self.conf.model.ssl_dim)
            stop_token_loss = torch.sum(stop_token_loss * stop_token_padding_mask) / torch.sum(stop_token_padding_mask)
            if token_loss is not None:
                token_loss = torch.sum(token_loss * token_padding_mask) / torch.sum(token_padding_mask)

        elif reduction == "utterance":
            if likelihood:
                # remove the padding frame
                log_likelihood, res = compute_log_likelihood(net=self.loss_fn.net, x=ssl_feats, z=logits)
                ll_abs_len = torch.round(wav_len * ssl_feats.shape[1]).long()
                ll_padding_mask = length_to_mask(ll_abs_len, max_len=log_likelihood.shape[1], dtype=torch.bool)
                # use BPD
                log_likelihood = torch.sum(log_likelihood * ll_padding_mask, dim=1) / torch.sum(ll_padding_mask, dim=1)
            loss = (torch.sum(loss * padding_mask, dim=1) / torch.sum(padding_mask, dim=1)).mean(dim=1)
            stop_token_loss = torch.sum(stop_token_loss * stop_token_padding_mask, dim=1) / torch.sum(stop_token_padding_mask, dim=1)
            if token_loss is not None:
                if self.args.ignore_eos and not self.training:
                    eos_index = self.gslm_pipeline.eos_token_index
                    L = token_padding_mask.shape[1]
                    token_padding_mask = token_padding_mask * (tokens[:, :L].squeeze(dim=2) != eos_index)
                token_loss = torch.sum(token_loss * token_padding_mask, dim=1) / torch.sum(token_padding_mask, dim=1)

        elif reduction == "unnormalized_utterance":
            if likelihood:
                # remove the padding frame
                log_likelihood, res = compute_log_likelihood(net=self.loss_fn.net, x=ssl_feats, z=logits)
                ll_padding_mask = padding_mask.squeeze(2)
                log_likelihood = torch.sum(log_likelihood * ll_padding_mask, dim=1)
            loss = torch.sum(loss * padding_mask, dim=1).sum(dim=1)
            stop_token_loss = torch.sum(stop_token_loss * stop_token_padding_mask, dim=1)
            if token_loss is not None:
                token_loss = torch.sum(token_loss * token_padding_mask, dim=1)

        total_loss = self.conf.optimizer.loss_weight * loss
        if self.conf.optimizer.token_loss_weight > 0:
            total_loss += self.conf.optimizer.token_loss_weight * token_loss
        if self.conf.optimizer.stop_token_weight > 0:
            total_loss += self.conf.optimizer.stop_token_weight * stop_token_loss 
            stop_token_f1 = self.get_stop_token_f1(is_stop_token, stop_token_target, stop_token_padding_mask)
        else:
            stop_token_f1 = 0
        if self.conf.optimizer.token_loss_weight > 0:
            if hasattr(self.conf.model, "extra_future_tokens") and self.conf.model.extra_future_tokens > 0:
                # monitoring the first token for now 
                token_logits_i = torch.chunk(token_logits, self.conf.model.extra_future_tokens, dim=2)[0]
                first_token_target = tokens[:, :-self.conf.model.extra_future_tokens + 1]
                token_acc = torch.sum((torch.argmax(token_logits_i, dim=-1) == first_token_target.reshape(first_token_target.shape[0], first_token_target.shape[1] * first_token_target.shape[2])).float() * token_padding_mask) / torch.sum(token_padding_mask)
            else:
                token_acc = torch.sum((torch.argmax(token_logits, dim=-1) == tokens.reshape(tokens.shape[0], tokens.shape[1] * tokens.shape[2])).float() * token_padding_mask) / torch.sum(token_padding_mask)
        else:
            token_acc = None

        if not likelihood:
            return total_loss, loss, stop_token_loss, stop_token_f1, token_loss, token_acc
        else:
            return log_likelihood, total_loss, loss, token_loss, token_acc

    def get_stop_token_f1(self, is_stop_token, stop_token_target, padding_mask):
        padding_mask = padding_mask.bool()
        pred = (is_stop_token > 0)[padding_mask].int()
        target = stop_token_target[padding_mask].int()
        f1 = binary_f1_score(pred, target)
        return f1

    def training_step(self, batch, batch_idx):
        total_loss, loss, stop_token_loss, stop_token_f1, token_loss, token_acc = self.forward(batch)
        current_lr = self.optimizers().param_groups[0]['lr']
        # skip the batch if nan
        if torch.isnan(total_loss):
            print("nan detected! skip this batch")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        if token_loss is not None:
            self.log("train/token_loss", token_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            self.log("train/token_acc", token_acc, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/stop_token_loss", stop_token_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/stop_token_f1", stop_token_f1, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx): 
        total_loss, loss, stop_token_loss, stop_token_f1, token_loss, token_acc = self.forward(batch)
        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if token_loss is not None:
            self.log("valid/token_loss", token_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("valid/token_acc", token_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("valid/stop_token_loss", stop_token_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("valid/stop_token_f1", stop_token_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def predict_step(self, batch, batch_idx):
        # same as validation step for now
        with torch.enable_grad():
            ids, wavs, wav_len = batch
            if self.args.compute_log_likelihood:
                log_likelihood, total_loss, loss, token_loss, token_acc = self.forward(batch, reduction=self.args.reduction, likelihood=True)
            else:
                total_loss, loss, stop_token_loss, stop_token_f1, token_loss, token_acc = self.forward(batch, reduction=self.args.reduction)

        # use score instead of loss
        if self.args.compute_log_likelihood and token_loss is not None:
            return ids, log_likelihood, -loss, -token_loss
        elif self.args.compute_log_likelihood:
            return ids, log_likelihood, -loss
        elif token_loss is not None:
            return ids, -loss, -token_loss
        else:
            return ids, -loss

    def test_step(self, batch, batch_idx):
        # same as validation step for now
        total_loss, loss, token_loss, token_acc = self.forward(batch, reduction='token')
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if token_loss is not None:
            self.log("test/token_loss", token_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("test/token_acc", token_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss, token_loss, token_acc
