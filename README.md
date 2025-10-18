# flow-slm

This repository contains code to train and run inference with the Flow-SLMs.

1. Install dependencies
-----------------------

Quick install (recommended: create and activate a conda environment first):

```bash
# create a conda env (optional)
conda create -n flow-slm python=3.10 -y
conda activate flow-slm

# Install requirements (pick torch+torchaudio matching your CUDA from https://pytorch.org)
pip install -r requirements.txt
```

Notes:
- If you plan to use the 8-bit optimizer (`AdamW8bit`), install `bitsandbytes` that matches your CUDA and PyTorch.
- Whisper-based ASR requires `whisper` and `ffmpeg` on PATH.

2. Training
-----------

Main training script: `trainer.py`.

Basic local training example:

```bash
python trainer.py --conf conf/<config>.yaml --save_path /path/to/ckpt_dir --hf_training_data --training_data "MLSEn+people"
```

Key options:
- `--conf`: path to YAML config under `conf/`.
- `--save_path`: directory where checkpoints and logs will be written.
- `--hf_training_data`: use HuggingFace datasets loader.
- `--training_data`: dataset shorthand used by the datamodule (e.g., `MLSEn+people`).
- `--override '<dict>'`: override config values at runtime (example below).


```bash
python trainer.py \
	--conf conf/1b_extended.yaml \
	--save_path /share/data/speech/ckpt/test \
	--override "{'optimizer': {'lr': 2e-4, 'loss_function': 'FM'}, 'training': {'batch_size': 8}}" \
	--hf_training_data --training_data "MLSEn+people" \
	--strategy "deepspeed_stage_3"
```

3. Inference
------------

Checkpoints 
-------------------
| Name | Description | Link |
|------|-------------|------:|
| Flow-SLM-270M | Trained on MLS (45k hours) | https://drive.google.com/file/d/1j9Gj39T-9lPN_ebGZ_xCGu9W3SQphjEN/view?usp=drive_link |
| Flow-SLM-1B | Trained on MLS (45k hours) | https://drive.google.com/file/d/1lh2JSNt3NUn--3uQscwteD-5YQlmno5z/view?usp=drive_link |
| Flow-SLM-1B-extend | Trained on MLS (45k hours) + People's Speech clean subset (20k) | https://drive.google.com/file/d/1YKiv-BD5r3MoCUZemHSWbZFv_diE2AbH/view?usp=drive_link |

Main inference script: `inference.py`.

Helper scripts:
- `prompt_inference.sh` - example for inference.

Important inference notes:
- To transcribe generated audio with Whisper, install `whisper` and optionally set `--download_whisper_root` to cache models.

4. Citation
-----------
```
@article{chou2025flow,
  title={Flow-slm: Joint learning of linguistic and acoustic information for spoken language modeling},
  author={Chou, Ju-Chieh and Zhou, Jiawei and Livescu, Karen},
  journal={arXiv preprint arXiv:2508.09350},
  year={2025}
}
```
