from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from datasets import concatenate_datasets
import torchaudio
import os
from datasets import load_dataset
from utils import batch_pad_right
import random
import torch
from typing import Optional, Tuple, List, Sequence

random.seed(0)

class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, args, conf):
        super().__init__()
        self.args = args
        self.conf = conf

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            print("set up datasets...")
            print(f"using {self.args.training_data} training data")

            if self.args.training_data == "MLSEn10k":
                size = "10k"
            elif self.args.training_data in ("MLSEn", "MLSEn+people"):
                size = "full"
            else:
                raise ValueError(f"{self.args.training_data} is not supported")

            vad = getattr(self.conf.data, "vad", False)

            if self.args.training_data == "MLSEn+people":
                mls_train_set = HFListDataset(kind="mls", size=size, split="train", vad=vad)
                people_train_set = HFListDataset(kind="people", split="train")
                self.train_set = torch.utils.data.ConcatDataset([mls_train_set, people_train_set])
            else:
                self.train_set = HFListDataset(kind="mls", size=size, split="train", vad=vad)

            self.val_set = HFListDataset(kind="mls", size=size, split="dev", vad=vad)

        if stage in ("predict", "test"):
            self.test_set = SpeechDataset(self.args.predict_id_file, self.args.data_dir,
                                          default_sr=self.conf.data.sr, ext=self.conf.data.ext)

    def train_dataloader(self):
        print("getting training loader...")
        return get_dataloader(self.train_set, batch_size=self.conf.training.batch_size,
                              shuffle=True, num_workers=self.conf.training.num_workers)

    def val_dataloader(self):
        print("getting validation loader...")
        return get_dataloader(self.val_set, batch_size=self.conf.training.batch_size,
                              shuffle=False, num_workers=self.conf.training.num_workers)

    def test_dataloader(self):
        print("getting test loader...")
        return get_dataloader(self.test_set, batch_size=self.conf.training.batch_size,
                              shuffle=False, num_workers=self.conf.training.num_workers, prefetch_factor=None)

    def predict_dataloader(self):
        print("getting predict loader...")
        return get_dataloader(self.test_set, batch_size=self.conf.training.batch_size,
                              shuffle=False, prefetch_factor=None)


class HFListDataset(Dataset):
    """
    Unified wrapper for HuggingFace datasets used in this repo:
      - kind="mls": uses parler-tts/mls_eng or parler-tts/mls_eng_10k
      - kind="people": uses MLCommons/peoples_speech with ['clean','clean_sa'] concatenated
    Returns tuples: (id, torch.Tensor(wav))
    """

    def __init__(self, kind: str = "mls", size: str = "full", pad_audio: bool = False,
                 reduction: int = 4, sort: bool = False, split: str = "train",
                 vad: bool = False):
        super().__init__()
        self.kind = kind
        self.vad = vad
        self.sort = sort
        self.split = split

        if kind == "people":
            datasets = [load_dataset("MLCommons/peoples_speech", subset, split=split)
                        for subset in ("clean", "clean_sa")]
            self.dataset = concatenate_datasets(datasets)
        elif kind == "mls":
            if size == "10k":
                self.dataset = load_dataset("parler-tts/mls_eng_10k", split=split)
            else:
                self.dataset = load_dataset("parler-tts/mls_eng", split=split)
        else:
            raise ValueError(f"Unknown kind {kind}")

        if self.sort:
            # sort in-place by audio duration if key exists
            if "audio_duration" in self.dataset.column_names:
                self.dataset = self.dataset.sort("audio_duration")

        if self.vad:
            self.vad_transform = torchaudio.transforms.Vad(sample_rate=16000)


    def __len__(self):
        return len(self.dataset)

    def _apply_vad(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.vad:
            return wav
        if wav.shape[0] <= 16000:
            return wav

        trimmed = self.vad_transform(wav)
        # try trimming from both ends
        if trimmed.shape[0] > 0.5 * wav.shape[0]:
            rev = torch.flip(trimmed, [0])
            rev = self.vad_transform(rev)
            trimmed = torch.flip(rev, [0])
        if trimmed.shape[0] > 0.5 * wav.shape[0]:
            return trimmed
        return wav

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        row = self.dataset[index]
        wav = torch.tensor(row["audio"]["array"], dtype=torch.float32)

        wav = self._apply_vad(wav)

        if self.kind == "people":
            uid = f"people_{row['id']}"
        else:
            uid = f"{row['original_path']}_{row['begin_time']}_{row['end_time']}_{row['book_id']}_{row['speaker_id']}"
        return uid, wav


class SpeechDataset(Dataset):
    """
    A simple file-based dataset driven by an id file (one id per line).
    Expects files in data_dir with names <id>.<ext>.
    """

    def __init__(self, id_file: str, data_dir: str, default_sr: int = 16000,
                 ext: str = "wav"):
        self.default_sr = default_sr
        self.data_dir = data_dir

        with open(id_file, "r") as fh:
            self.ids: List[str] = [line.strip() for line in fh if line.strip()]

        self.wav_list: List[str] = [os.path.join(self.data_dir, f"{uid}.{ext}") for uid in self.ids]

    def __len__(self) -> int:
        return len(self.wav_list)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        uid = self.ids[index]
        path = self.wav_list[index]
        wav, sr = torchaudio.load(path)
        if wav.dim() == 2:
            wav = wav[0]
        if sr != self.default_sr:
            wav = torchaudio.functional.resample(wav, sr, self.default_sr)
        return uid, wav


class Collator:
    def __init__(self):
        pass

    def wav_collate_fn(self, batch: Sequence[Tuple[str, torch.Tensor]]):
        ids = [entry[0] for entry in batch]
        wavs = [entry[1] for entry in batch]
        wavs, wav_len = batch_pad_right(wavs)
        return ids, wavs, wav_len


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, batch_sampler=None,
                   drop_last: bool = True, num_workers: int = 0, prefetch_factor: Optional[int] = 2) -> DataLoader:
    collator = Collator()
    collate_fn = collator.wav_collate_fn

    if batch_sampler is None:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=num_workers, collate_fn=collate_fn,
                                 drop_last=drop_last, pin_memory=True, prefetch_factor=prefetch_factor)
    else:
        data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                                 num_workers=num_workers, collate_fn=collate_fn,
                                 pin_memory=True, prefetch_factor=prefetch_factor)
    return data_loader

