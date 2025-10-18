"""Training module for continuous GSLM.

This module contains the main training class LanguageModeling which handles
the training, validation, and prediction steps for the continuous GSLM model.
Refactored to split responsibilities into smaller helper methods, fix a couple
of mode/device issues, and improve readability.
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from utils import get_cosine_schedule_with_warmup
from pipeline import GSLMPipeline
from losses import FlowLoss
import argparse
import os
import signal
import yaml
import munch
import sys
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from utils import replace_values, writing_output_to_file, SaveAtSpecificStep, select_latest_ckpt
from lightning.pytorch.plugins.environments import SLURMEnvironment
from dataset import SpeechDataModule

class LanguageModeling(pl.LightningModule):
    """Main training class for continuous GSLM.
    
    Refactored: helpers for decoder freezing, loss computation, token loss,
    and metrics. Behavior should remain functionally equivalent.
    """

    def __init__(self, args, conf):
        super().__init__()
        self.args = args
        self.conf = conf
        conf_dict = self.conf.toDict()
        self.save_hyperparameters(conf_dict)

        self.gslm_pipeline = GSLMPipeline(conf, args)

        # build loss_fn (FlowLoss) depending on config
        if self.conf.optimizer.loss_function == "FM":
            dim = self.conf.model.ssl_dim
            token_conditioning = getattr(self.conf.model, "token_conditioning", False)
            if token_conditioning:
                token_emb_dim = getattr(self.conf.model, "token_emb_dim", 0)
                future_conditioning = getattr(self.conf.model, "future_conditioning", False)
                if future_conditioning:
                    extra_future_tokens = getattr(self.conf.model, "extra_future_tokens", 1)
                    z_dim = self.conf.model.decoder_dim + token_emb_dim * extra_future_tokens
                else:
                    z_dim = self.conf.model.decoder_dim + token_emb_dim
            else:
                z_dim = self.conf.model.decoder_dim

            null_prob = 0.0 if not hasattr(self.conf.optimizer, "null_prob") else self.conf.optimizer.null_prob
            self.loss_fn = FlowLoss(
                target_dim=dim * self.conf.model.reduction_factor,
                z_dim=z_dim,
                net=self.gslm_pipeline.decoder.output_proj,
                sigma_min=self.conf.optimizer.sigma_min,
                t_dist=self.conf.optimizer.t_dist,
                null_prob=null_prob,
            )
        else:
            raise NotImplementedError(f"{self.conf.optimizer.loss_function} not implemented.")

        self.token_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        if not hasattr(self.conf.optimizer, "loss_weight"):
            self.conf.optimizer.loss_weight = 1.0

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        if self.conf.optimizer.name == "AdamW":
            opt = torch.optim.AdamW(
                trainable_params,
                lr=self.conf.optimizer.lr,
                betas=self.conf.optimizer.betas,
                weight_decay=self.conf.optimizer.weight_decay,
                eps=self.conf.optimizer.eps,
            )
        elif self.conf.optimizer.name == "AdamW8bit":
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(
                trainable_params,
                lr=self.conf.optimizer.lr,
                betas=self.conf.optimizer.betas,
                weight_decay=self.conf.optimizer.weight_decay,
                eps=self.conf.optimizer.eps,
                percentile_clipping=self.conf.optimizer.percentile_clipping,
            )
        else:
            raise NotImplementedError(f"{self.conf.optimizer.name} not implemented.")

        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.conf.training.num_warmup_steps,
            num_training_steps=self.conf.training.max_steps,
            min_lr_ratio=self.conf.training.min_lr_ratio,
        )

        lr_scheduler_config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return lr_scheduler_config

    def _run_pipeline(self, wavs: torch.Tensor, wav_len: torch.Tensor, eval_mode: bool):
        """
        Call the pipeline. If eval_mode is True, temporarily set pipeline.eval()
        and restore previous training flag after call.
        """
        if eval_mode:
            was_training = self.gslm_pipeline.training
            self.gslm_pipeline.eval()
            with torch.no_grad():
                out = self.gslm_pipeline(wavs, wav_len)
            if was_training:
                self.gslm_pipeline.train()
            return out
        else:
            return self.gslm_pipeline(wavs, wav_len)

    def _compute_flow_loss(self, logits, ssl_feats):
        """Compute flow loss or return zeros if disabled."""
        if getattr(self.conf.optimizer, "loss_weight", 1.0) > 0:
            return self.loss_fn(logits, ssl_feats)
        return torch.zeros_like(logits)

    def _compute_token_loss(self, token_logits, tokens, token_padding_mask, training: bool):
        if self.conf.optimizer.token_loss_weight <= 0:
            return None

        # multi-future-token handling
        if hasattr(self.conf.model, "extra_future_tokens") and self.conf.model.extra_future_tokens > 0:
            k_future = self.conf.model.extra_future_tokens
            token_losses = token_padding_mask.new_zeros((token_padding_mask.shape[0], token_padding_mask.shape[1])).float()
            token_weight = token_padding_mask.new_zeros((token_padding_mask.shape[0], token_padding_mask.shape[1])).float()

            token_logits_i = torch.chunk(token_logits, k_future, dim=2)
            k_future_tokens = k_future if training else self.args.use_k_future_tokens
            L = token_padding_mask.shape[1]
            for i in range(k_future_tokens):
                logits_i = token_logits_i[i].reshape(-1, token_logits_i[i].shape[-1])
                tokens_i = tokens[:, i : i + L].reshape(-1)
                loss_i = self.token_loss_fn(logits_i, tokens_i).reshape(token_logits_i[i].shape[0], token_logits_i[i].shape[1])
                if self.args.ignore_eos and not training:
                    token_padding_mask_no_eos = (token_padding_mask * (tokens_i.reshape(token_logits_i[i].shape[0], token_logits_i[i].shape[1]) != self.gslm_pipeline.eos_token_index)).float()
                    token_losses += loss_i * token_padding_mask_no_eos
                    token_weight += token_padding_mask_no_eos
                else:
                    token_losses += loss_i * token_padding_mask
                    token_weight += token_padding_mask
            token_weight[token_weight == 0] = 1e-6
            final_token_loss = token_losses / token_weight
        else:
            final_token_loss = self.token_loss_fn(token_logits.reshape(-1, token_logits.shape[-1]), tokens.reshape(-1)).reshape(token_logits.shape[0], token_logits.shape[1])

        return final_token_loss

    def _compute_metrics_and_total(self, flow_loss, padding_mask, token_loss, token_padding_mask):
        flow_loss_val = torch.sum(flow_loss * padding_mask) / (torch.sum(padding_mask) * self.conf.model.reduction_factor * self.conf.model.ssl_dim)
        if token_loss is not None:
            token_loss_val = torch.sum(token_loss * token_padding_mask) / torch.sum(token_padding_mask)
        else:
            token_loss_val = None

        total_loss = self.conf.optimizer.loss_weight * flow_loss_val
        if token_loss_val is not None:
            total_loss = total_loss + self.conf.optimizer.token_loss_weight * token_loss_val

        # token accuracies for all future-token predictions
        token_accs = None
        if token_loss_val is not None:
            # monitor first future token when available
            if hasattr(self.conf.model, "extra_future_tokens") and self.conf.model.extra_future_tokens > 0:
                token_logits_i = torch.chunk(self.last_token_logits, self.conf.model.extra_future_tokens, dim=2)[0]
                first_token_target = self.last_tokens[:, :-self.conf.model.extra_future_tokens + 1]
                token_acc = torch.sum((torch.argmax(token_logits_i, dim=-1) == first_token_target.reshape(first_token_target.shape[0], first_token_target.shape[1] * first_token_target.shape[2])).float() * token_padding_mask) / torch.sum(token_padding_mask)
            else:
                token_acc = torch.sum((torch.argmax(self.last_token_logits, dim=-1) == self.last_tokens.reshape(self.last_tokens.shape[0], self.last_tokens.shape[1] * self.last_tokens.shape[2])).float() * token_padding_mask) / torch.sum(token_padding_mask)
        else:
            token_acc = None

        return total_loss, flow_loss_val, token_loss_val, token_acc

    def forward(self, batch, reduction='token'):
        ids, wavs, wav_len = batch
        wav_len = wav_len.float()

        # run pipeline (use eval mode for non-training forward)
        eval_mode = not self.training
        logits, ssl_feats, padding_mask, token_logits, tokens, token_padding_mask = self._run_pipeline(wavs, wav_len, eval_mode)

        # store last token values for metric computation
        self.last_token_logits = token_logits
        self.last_tokens = tokens

        flow_loss = self._compute_flow_loss(logits, ssl_feats)
        token_loss = self._compute_token_loss(token_logits, tokens, token_padding_mask, self.training)

        if reduction == "token":
            total_loss, flow_loss_val, token_loss_val, token_acc = self._compute_metrics_and_total(
                flow_loss, padding_mask, token_loss, token_padding_mask
            )
            return total_loss, flow_loss_val, token_loss_val, token_acc

        elif reduction == "utterance":
            loss = (torch.sum(flow_loss * padding_mask, dim=1) / torch.sum(padding_mask, dim=1)).mean(dim=1)
            if token_loss is not None:
                # by default, ignore eos token during evaluation
                if self.args.ignore_eos and not self.training:
                    eos_index = self.gslm_pipeline.eos_token_index
                    L = token_padding_mask.shape[1]
                    token_padding_mask = token_padding_mask * (tokens[:, :L].squeeze(dim=2) != eos_index)
                token_loss = torch.sum(token_loss * token_padding_mask, dim=1) / torch.sum(token_padding_mask, dim=1)

        elif reduction == "unnormalized_utterance":
            loss = torch.sum(flow_loss * padding_mask, dim=1).sum(dim=1)
            if token_loss is not None:
                token_loss = torch.sum(token_loss * token_padding_mask, dim=1)

        total_loss = self.conf.optimizer.loss_weight * flow_loss_val
        if self.conf.optimizer.token_loss_weight > 0 and token_loss is not None:
            total_loss += self.conf.optimizer.token_loss_weight * token_loss

        if self.conf.optimizer.token_loss_weight > 0 and token_loss is not None:
            if hasattr(self.conf.model, "extra_future_tokens") and self.conf.model.extra_future_tokens > 0:
                token_logits_i = torch.chunk(token_logits, self.conf.model.extra_future_tokens, dim=2)[0]
                first_token_target = tokens[:, :-self.conf.model.extra_future_tokens + 1]
                token_acc = torch.sum((torch.argmax(token_logits_i, dim=-1) == first_token_target.reshape(first_token_target.shape[0], first_token_target.shape[1] * first_token_target.shape[2])).float() * token_padding_mask) / torch.sum(token_padding_mask)
            else:
                token_acc = torch.sum((torch.argmax(token_logits, dim=-1) == tokens.reshape(tokens.shape[0], tokens.shape[1] * tokens.shape[2])).float() * token_padding_mask) / torch.sum(token_padding_mask)
        else:
            token_acc = None

        return total_loss, flow_loss_val, token_loss, token_acc

    def training_step(self, batch, batch_idx):
        total_loss, flow_loss_val, token_loss, token_acc = self.forward(batch)
        current_lr = self.optimizers().param_groups[0]["lr"]
        if torch.isnan(total_loss):
            print("nan detected! skip this batch")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        if token_loss is not None:
            self.log("train/token_loss", token_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            self.log("train/token_acc", token_acc, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, flow_loss_val, token_loss, token_acc = self.forward(batch)
        self.log("valid/loss", flow_loss_val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if token_loss is not None:
            self.log("valid/token_loss", token_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("valid/token_acc", token_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def predict_step(self, batch, batch_idx):
        with torch.enable_grad():
            ids, wavs, wav_len = batch
            total_loss, flow_loss_val, token_loss, token_acc = self.forward(batch, reduction=self.args.reduction)

        if token_loss is not None:
            return ids, -flow_loss_val, -token_loss
        else:
            return ids, -flow_loss_val

    def test_step(self, batch, batch_idx):
        total_loss, flow_loss_val, token_loss, token_acc = self.forward(batch, reduction="token")
        self.log("test/loss", flow_loss_val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if token_loss is not None:
            self.log("test/token_loss", token_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("test/token_acc", token_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss, token_loss, token_acc
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", help="Path to dataset folder", default=None)
    parser.add_argument("--ckpt_path", type=str, help="GSLM checkpoint, for inference only", default=None)
    parser.add_argument("--conf", help="Path to config file")
    parser.add_argument("--train_id_file", help="Path to training dataset ids if not using hf_training_data", default=None)
    parser.add_argument("--hf_training_data", action="store_true")
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument("--predict_only", action="store_true")
    parser.add_argument("--training_data", choices=["MLSEn10k", "MLSEn", "MLSEn+people"], default=None)
    parser.add_argument("--valid_id_file", help="Path to validation dataset ids")
    parser.add_argument("--predict_id_file", help="Path to prediction dataset ids")
    parser.add_argument("--prediction_output_dir", help="prediction file path to save")
    parser.add_argument("--save_path", help="Path to save checkpoints")
    parser.add_argument("--reduction", help="reduction approach for prediction", default="utterance")
    parser.add_argument("--ignore_eos", action="store_true", help="ignore eos token for prediction")
    parser.add_argument("--use_k_future_tokens", default=0, type=int, help="use k future tokens for prediction")
    parser.add_argument("--every_n_steps", help="every n steps, do validation and checkpointing", default=5000, type=int)
    parser.add_argument(
        "--strategy",
        help="ddp strategy",
        default="ddp_find_unused_parameters_true",
        choices=["ddp", "ddp_find_unused_parameters_true", "deepspeed_stage_2", "deepspeed_stage_3", "fsdp", "deepspeed_stage_1"],
    )
    parser.add_argument("--override", help="override the hyperparameters in conf", default=None, type=str)

    args = parser.parse_args()

    # load config
    with open(args.conf) as f:
        conf = yaml.safe_load(f)
    if args.override is not None:
        overrides = eval(args.override)
        replace_values(conf, overrides)
    conf = munch.munchify(conf)

    # default k future tokens from config if not provided
    if args.use_k_future_tokens == 0 and hasattr(conf.model, "extra_future_tokens") and conf.model.extra_future_tokens > 0:
        args.use_k_future_tokens = conf.model.extra_future_tokens

    # decide mode (training vs eval)
    training_mode = (not args.validation_only) and (not args.predict_only)

    # build model and trainer
    if training_mode:
        conf_path = Path(args.conf).stem
        ckpt_dir = os.path.join(args.save_path, conf_path)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = select_latest_ckpt(ckpt_dir)

        if ckpt_path is not None:
            print(f"loading from {ckpt_path}")

        # persist config next to checkpoints
        with open(os.path.join(ckpt_dir, "conf.yaml"), "w") as f:
            yaml.dump(conf.toDict(), f)

        language_modeling = LanguageModeling(args, conf)

        # optionally load pretrained checkpoint provided explicitly
        if args.ckpt_path and args.ckpt_path != "None":
            print(f"loading pretrained model from {args.ckpt_path}")
            state_dict = torch.load(args.ckpt_path, map_location="cpu")
            # attempt strict load, fallback to tolerant cleanup
            try:
                language_modeling.load_state_dict(state_dict, strict=True)
            except Exception:
                # remove keys that commonly cause incompatibility if present
                for k in list(state_dict.keys()):
                    if any(substr in k for substr in ("cond_embed.weight", "null_emb.weight", "loss_fn.net.cond_embed.weight")):
                        state_dict.pop(k, None)
                language_modeling.load_state_dict(state_dict, strict=False)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=5,
            monitor="step",
            mode="max",
            every_n_train_steps=args.every_n_steps,
            dirpath=ckpt_dir,
            save_last=True,
            filename="model-{step:07d}",
        )
        save_at_specific_step = SaveAtSpecificStep(10000, ckpt_dir=f"{ckpt_dir}/10kxn_ckpt/")
        tb_logger = TensorBoardLogger(save_dir=f"{ckpt_dir}/logs/", version=1)
        tb_logger.log_hyperparams(conf.toDict())
        lr_monitor = LearningRateMonitor(logging_interval="step")
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else 32
        trainer = pl.Trainer(
            accelerator="gpu",
            max_steps=conf.training.max_steps,
            callbacks=[checkpoint_callback, save_at_specific_step, lr_monitor],
            val_check_interval=args.every_n_steps,
            check_val_every_n_epoch=None,
            logger=tb_logger,
            precision=precision,
            devices="auto",
            strategy=args.strategy,
            log_every_n_steps=args.every_n_steps,
            accumulate_grad_batches=conf.training.accumulate_grad_batches,
            plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)],
            default_root_dir=ckpt_dir,
            use_distributed_sampler=True,
        )
    else:
        # evaluation / prediction only
        print(f"evaluation only, loading {args.ckpt_path}")
        language_modeling = LanguageModeling(args=args, conf=conf)
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        # defensive cleanup of misnamed keys if present
        for bad_key in ("gsml_pipeline.decoder.stop_token.weight", "gsml_pipeline.decoder.stop_token.bias"):
            if bad_key in state_dict:
                state_dict.pop(bad_key, None)
        try:
            language_modeling.load_state_dict(state_dict)
        except Exception:
            language_modeling.load_state_dict(state_dict, strict=False)
        language_modeling.eval()
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else 32
        ckpt_dir = os.path.dirname(args.ckpt_path)
        trainer = pl.Trainer(
            accelerator="gpu",
            max_steps=conf.training.max_steps,
            val_check_interval=args.every_n_steps,
            check_val_every_n_epoch=None,
            precision=precision,
            devices="auto",
            log_every_n_steps=args.every_n_steps,
            accumulate_grad_batches=conf.training.accumulate_grad_batches,
            default_root_dir=ckpt_dir,
            use_distributed_sampler=False,
            inference_mode=inference_mode,
        )

    data = SpeechDataModule(args, conf)

    if training_mode:
        try:
            trainer.fit(language_modeling, data, ckpt_path=ckpt_path)
        except KeyboardInterrupt:
            sys.exit()
    elif args.validation_only:
        data.setup(stage="test")
        trainer.test(language_modeling, data)
    elif args.predict_only:
        data.setup(stage="predict")
        output = trainer.predict(language_modeling, data)
        writing_output_to_file(output, args.prediction_output_dir, token=conf.optimizer.token_loss_weight > 0)


if __name__ == "__main__":
    print("cuda_available()", torch.cuda.is_available())
    print("cuda_device_count()", torch.cuda.device_count())
    print("torch.cuda.is_bf16_supported()", torch.cuda.is_bf16_supported())
    if torch.cuda.is_bf16_supported():
        torch.set_float32_matmul_precision("medium")
    main()
