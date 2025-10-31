import argparse
import csv
from pathlib import Path
import tqdm

import lightning.pytorch as pl
import munch
import numpy as np
import torch
import torchaudio
import yaml

from decode import Sampler
from trainer import LanguageModeling
from model_utils import reduce_features
from dataset import get_dataloader
from model import MimiDecoder

try:
    import whisper
except Exception:
    whisper = None

class WhisperWrapper:
    def __init__(self, model_card="small.en", device="cuda", resample=True, download_root=None):
        if whisper is None:
            raise RuntimeError("whisper not available")
        self.device = device
        self.model = whisper.load_model(model_card, download_root=download_root, device=device)
        self.options = whisper.DecodingOptions(language="en", without_timestamps=True)

    def transcribe(self, audio):
        audio = whisper.pad_or_trim(audio).to(self.device)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels)
        result = self.model.decode(mel, self.options)[0]
        return result


def load_audio_list(root_dir: str, csv_path: str, target_sample_rate: int):
    data = []
    root_dir = Path(root_dir)
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            audio_path = root_dir / row["path"]
            duration = float(row.get("prompt_length", 0.0))
            waveform, sample_rate = torchaudio.load(str(audio_path))
            if sample_rate != target_sample_rate:
                waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
            prompt_id = row["path"].replace("/", "_").replace(".wav", "").replace(".flac", "")
            data.append((prompt_id, waveform, duration))

    return data


class Processor:
    def __init__(self, conf, device="cuda"):
        self.count = 0
        self.conf = conf
        self.device = device

        if getattr(self.conf.model, "norm", None) == "static":
            self.mean = torch.Tensor(np.load(conf.model.mean_path)).to(self.device)
            self.std = torch.Tensor(np.load(conf.model.std_path)).to(self.device)

        self.frame_rate = 12.5

        self.vocoder = None
        self.sample_rate = None

    def load_ssl_model(self, ssl_model):
        self.ssl_model = ssl_model
        return

    def get_ssl_feats(self, wav, duration, duplicate=1):
        wavs = wav.to(self.device)
        wav_lens = torch.ones(1).to(self.device)

        with torch.no_grad():
            if getattr(self.ssl_model, "n_quantizers", 0) > 0:
                feats, codes = self.ssl_model(wavs, wav_lens)
            else:
                feats = self.ssl_model(wavs, wav_lens)

            if getattr(self.conf.model, "norm", None) == "static":
                feats = (feats - self.mean) / self.std

            feats = feats[:, :int(duration * self.frame_rate)]

            reduced_feats = reduce_features(feats, self.conf.model.reduction_factor)
            reduced_feats = reduced_feats.repeat(duplicate, 1, 1)

            return reduced_feats

    def load_vocoder_mimi(self):
        self.vocoder_type = "mimi"
        self.vocoder = MimiDecoder().cuda()
        self.sample_rate = 24000
        return

    def batch_vocoding(self, samples, stop_steps=None, num_quantizers=None):
        self.count += 1
        lengths = stop_steps * self.conf.model.reduction_factor

        if self.vocoder_type == "mimi":
            hop_size = 1920
        else:
            raise NotImplementedError

        wav_lens = lengths * hop_size
        with torch.no_grad():
            batch_wavs = self.vocoder(samples, num_quantizers=num_quantizers)

        wavs = []
        for i, wav in enumerate(batch_wavs):
            wav = wav[: wav_lens[i]].unsqueeze(dim=0)
            wavs.append(wav)
        return wavs

    def unmerge_and_unnormalize(self, samples):
        bs, seq_len, feat_dim = samples.shape
        samples = samples.reshape(bs, seq_len * self.conf.model.reduction_factor, feat_dim // self.conf.model.reduction_factor)

        if getattr(self.conf.model, "norm", None) == "static":
            samples = samples * self.std + self.mean

        return samples


def parse_args():
    parser = argparse.ArgumentParser(description="GSLM inference")
    parser.add_argument("--ckpt_path", type=str, required=True, help="GSLM checkpoint")
    parser.add_argument("--conf_path", type=str, required=True, help="config file")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--samples_per_prompt", type=int, default=16)
    parser.add_argument("--prompt_dir", type=str, default=None)
    parser.add_argument("--prompt_csv", type=str, default=None)
    parser.add_argument("--min_len", type=float, default=10)
    parser.add_argument("--max_len", type=float, default=30)
    parser.add_argument("--ode_steps", type=int, default=32)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--topp", type=float, default=0.95)
    parser.add_argument("--penalize_silence", action="store_true")
    parser.add_argument("--penalize_weight", type=float, default=10.0)
    parser.add_argument("--token_temperature", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--cfg_scale", type=float, default=0.3)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--shift_alpha", type=float, default=1.0)
    parser.add_argument("--save_wav", action="store_true")
    parser.add_argument("--sample_with_gt_tokens", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--asr", action="store_true", help="Use Whisper to transcribe generated audio")
    parser.add_argument("--download_whisper_root", type=str, default=None, help="Root directory to download Whisper models")
    parser.add_argument("--save_transcription", action="store_true")
    parser.add_argument("--num_quantizers", type=int, default=16)
    return parser.parse_args()


def load_conf(conf_path):
    with open(conf_path) as f:
        conf = yaml.safe_load(f)
    conf = munch.munchify(conf)
    conf.model.flash_attention = False
    return conf


def load_model(args, conf, device="cuda"):
    model_args = type("Args", (), {})()
    lm = LanguageModeling(model_args, conf)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    lm.load_state_dict(state_dict)
    lm = lm.to(device).to(torch.bfloat16)
    print(lm)
    return lm


def prepare_sampler_and_processor(lm, conf, args, device="cuda"):
    frame_rate = 12.5
    sampler = Sampler(lm.gslm_pipeline, lm.loss_fn, frame_rate=frame_rate).to(device)
    processor = Processor(conf, device=device)
    processor.load_vocoder_mimi()
    processor.load_ssl_model(lm.gslm_pipeline.ssl_model.to(torch.float32))

    return sampler, processor

def save_wav(wav, path, sample_rate):
    torchaudio.save(path, wav.cpu(), sample_rate, backend="soundfile")

def run_unconditional(args, sampler, processor):
    codec_size = 2048
    samples_to_generate = args.n_samples
    batch_size = args.batch_size
    generated = 0
    while generated < samples_to_generate:
        cur_bs = min(batch_size, samples_to_generate - generated)
        with torch.no_grad():
            # eos_token is set to the last token id if token loss is used
            eos_aux_token = codec_size + processor.conf.model.n_special_tokens - 1 if getattr(processor.conf.optimizer, "token_loss_weight", 0) > 0 else None
            samples, stop_steps = sampler.sample(batch_size=cur_bs, min_len=args.min_len, max_len=args.max_len, ode_steps=args.ode_steps, token_temperature=args.token_temperature, temperature=args.temperature, solver=args.solver, eos_aux_token=eos_aux_token, cfg_scale=args.cfg_scale, topk=args.topk, topp=args.topp, penalize_silence=args.penalize_silence, penalize_weight=args.penalize_weight)

        samples = processor.unmerge_and_unnormalize(samples)
        wavs = processor.batch_vocoding(samples, stop_steps, args.num_quantizers)
        for wav in wavs:
            yield str(generated), wav, processor.sample_rate
        generated += cur_bs


def run_conditional(args, sampler, processor, prompt_wavs):
    codec_size = 2048
    for prompt_idx, (prompt_id, wav, duration) in enumerate(prompt_wavs):
        reduced_feats = processor.get_ssl_feats(wav, duration, duplicate=args.batch_size)
        reduced_feats = reduced_feats.to(torch.bfloat16)

        for batch_idx in range(args.samples_per_prompt // args.batch_size):
            with torch.no_grad():
                eos_aux_token = codec_size + processor.conf.model.n_special_tokens - 1 if getattr(processor.conf.optimizer, "token_loss_weight", 0) > 0 else None
                samples, stop_steps = sampler.sample(batch_size=args.batch_size, min_len=args.min_len, max_len=args.max_len, ode_steps=args.ode_steps, token_temperature=args.token_temperature, temperature=args.temperature, prompts=reduced_feats, solver=args.solver, eos_aux_token=eos_aux_token, cfg_scale=args.cfg_scale, topk=args.topk, topp=args.topp, penalize_silence=args.penalize_silence, penalize_weight=args.penalize_weight)

            samples = processor.unmerge_and_unnormalize(samples)
            wavs = processor.batch_vocoding(samples, stop_steps, args.num_quantizers)

            for i, wav in enumerate(wavs):
                yield prompt_id, wav, processor.sample_rate


def main():
    args = parse_args()
    conf = load_conf(args.conf_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading {args.ckpt_path}")
    lm = load_model(args, conf, device=device)
    lm.eval()

    sampler, processor = prepare_sampler_and_processor(lm, conf, args, device=device)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    transcription_file = None
    if args.save_transcription and args.output_dir:
        transcription_file = open(Path(args.output_dir) / "transcriptions.csv", "w")

    asr_model = None
    if args.asr:
        if whisper is None:
            print("Whisper not installed; ASR disabled.")
        else:
            asr_model = WhisperWrapper(model_card="large-v3-turbo", device=device, download_root=args.download_whisper_root)

    # load prompt list if provided
    prompt_wavs = None
    if args.prompt_csv is not None and args.prompt_dir is not None:
        prompt_wavs = load_audio_list(args.prompt_dir, args.prompt_csv, target_sample_rate=16000)

    if prompt_wavs is None:
        gen_iter = run_unconditional(args, sampler, processor)
    else:
        gen_iter = run_conditional(args, sampler, processor, prompt_wavs)

    for idx, (prompt_id, wav, sr) in enumerate(tqdm.tqdm(gen_iter)):
        if args.save_wav and args.output_dir:
            if prompt_wavs is not None:
                sample_idx = idx % args.samples_per_prompt
                out_path = Path(args.output_dir) / f"{prompt_id}_{sample_idx:04d}.wav"
            else:
                out_path = Path(args.output_dir) / f"{idx:04d}.wav"
            save_wav(wav, str(out_path), sr)

        if asr_model is not None:
            try:
                res = asr_model.transcribe(wav)
            except Exception as e:
                print("ASR failed:", e)
                res = None
        else:
            res = None

        if transcription_file is not None and res is not None:
            print(f"{prompt_id}_{sample_idx:04d}\t{res.text}", file=transcription_file, flush=True)

    if transcription_file is not None:
        transcription_file.close()


if __name__ == "__main__":
    main()
