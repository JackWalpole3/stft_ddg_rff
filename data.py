import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig


def power_normalize_iq(x: np.ndarray) -> np.ndarray:
    """Match the TFMix max-power normalization on IQ samples."""
    x = x.astype(np.float32, copy=True)
    power = np.square(x[:, :, 0]) + np.square(x[:, :, 1])
    scale = np.sqrt(np.maximum(power.max(axis=1, keepdims=True), 1e-12))
    return x / scale[:, :, None]


def _split_one_class(num_samples: int, cfg: ExperimentConfig, seed: int) -> Dict[str, np.ndarray]:
    needed = cfg.train_per_tx + cfg.val_per_tx + cfg.test_per_tx
    if num_samples < needed:
        raise ValueError(f"Need {needed} samples per tx, got {num_samples}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)[:needed]
    train_end = cfg.train_per_tx
    val_end = cfg.train_per_tx + cfg.val_per_tx
    return {
        "train": perm[:train_end],
        "val": perm[train_end:val_end],
        "test": perm[val_end:],
    }


def load_receiver_iq(rx: str, cfg: ExperimentConfig) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    with open(cfg.data_path, "rb") as f:
        data = pickle.load(f)
    rx_list = data["rx_list"]
    if rx not in rx_list:
        raise ValueError(f"Receiver {rx!r} not found in {rx_list}")
    rx_idx = rx_list.index(rx)

    splits_x: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    splits_y: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}

    for tx in range(cfg.num_classes):
        tx_data = data["data"][tx][rx_idx][cfg.day_idx][cfg.eq_idx]
        if tx_data.shape[0] < cfg.samples_per_tx:
            raise ValueError(
                f"Not enough samples for rx={rx}, tx={tx}: "
                f"{tx_data.shape[0]} < {cfg.samples_per_tx}"
            )
        tx_data = tx_data[: cfg.samples_per_tx]
        tx_seed = cfg.seed + 1009 * rx_idx + 97 * tx
        idx = _split_one_class(tx_data.shape[0], cfg, tx_seed)
        for split, split_idx in idx.items():
            splits_x[split].append(tx_data[split_idx])
            splits_y[split].append(np.full(split_idx.shape[0], tx, dtype=np.int64))

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    rx_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(rx))
    split_offsets = {"train": 11, "val": 23, "test": 37}
    for split in ("train", "val", "test"):
        x = power_normalize_iq(np.concatenate(splits_x[split], axis=0))
        y = np.concatenate(splits_y[split], axis=0)
        order = np.random.default_rng(cfg.seed + rx_seed + split_offsets[split]).permutation(len(y))
        out[split] = x[order], y[order]
    return out


def iq_to_stft_tensor(iq: np.ndarray, cfg: ExperimentConfig, batch_size: int = 512) -> torch.Tensor:
    """Convert (N, 256, 2) real IQ to (N, 2, F, T) STFT real/imag tensor."""
    tensors: List[torch.Tensor] = []
    window = torch.hann_window(cfg.win_length)
    for start in range(0, iq.shape[0], batch_size):
        chunk = torch.from_numpy(iq[start : start + batch_size]).float()
        complex_iq = torch.complex(chunk[:, :, 0], chunk[:, :, 1])
        stft = torch.stft(
            complex_iq,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=window,
            center=cfg.center,
            return_complex=True,
            onesided=False,
        )
        stft_ri = torch.view_as_real(stft).permute(0, 3, 1, 2).contiguous()
        tensors.append(stft_ri)
    return torch.cat(tensors, dim=0)


class STFTDataset(Dataset):
    def __init__(
        self,
        x_iq: np.ndarray,
        y: np.ndarray,
        cfg: ExperimentConfig,
        return_pos: bool = False,
        seed: int = 0,
    ) -> None:
        self.x = iq_to_stft_tensor(x_iq, cfg)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.return_pos = return_pos
        self.pos_indices = self._build_pos_indices(seed) if return_pos else None

    def _build_pos_indices(self, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        y_np = self.y.numpy()
        pos = np.empty(len(y_np), dtype=np.int64)
        for i, label in enumerate(y_np):
            candidates = np.where(y_np == label)[0]
            if len(candidates) == 1:
                pos[i] = i
                continue
            choice = i
            while choice == i:
                choice = int(rng.choice(candidates))
            pos[i] = choice
        return torch.from_numpy(pos)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]
        if not self.return_pos:
            return x, y
        assert self.pos_indices is not None
        return x, y, self.x[self.pos_indices[idx]]


@dataclass
class CRDatasets:
    target_rx: str
    source_train: List[Tuple[str, STFTDataset]]
    source_val: List[Tuple[str, STFTDataset]]
    target_test: STFTDataset


def build_cr_datasets(target_rx: str, cfg: ExperimentConfig, return_pos: bool) -> CRDatasets:
    source_train: List[Tuple[str, STFTDataset]] = []
    source_val: List[Tuple[str, STFTDataset]] = []
    target_test = None
    for rx in cfg.receivers:
        splits = load_receiver_iq(rx, cfg)
        if rx == target_rx:
            x_test, y_test = splits["test"]
            target_test = STFTDataset(x_test, y_test, cfg, return_pos=False, seed=cfg.seed)
        else:
            x_train, y_train = splits["train"]
            x_val, y_val = splits["val"]
            source_train.append(
                (
                    rx,
                    STFTDataset(
                        x_train,
                        y_train,
                        cfg,
                        return_pos=return_pos,
                        seed=cfg.seed + len(source_train),
                    ),
                )
            )
            source_val.append(
                (rx, STFTDataset(x_val, y_val, cfg, return_pos=False, seed=cfg.seed))
            )
    if target_test is None:
        raise ValueError(f"Target receiver {target_rx!r} was not built")
    return CRDatasets(target_rx, source_train, source_val, target_test)


def make_loader(dataset: Dataset, cfg: ExperimentConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def make_domain_loaders(
    domain_datasets: Sequence[Tuple[str, STFTDataset]],
    cfg: ExperimentConfig,
    shuffle: bool,
) -> List[Tuple[str, DataLoader]]:
    return [(rx, make_loader(ds, cfg, shuffle=shuffle)) for rx, ds in domain_datasets]


def count_by_class(dataset: STFTDataset, num_classes: int) -> List[int]:
    y = dataset.y.numpy()
    return [int((y == cls).sum()) for cls in range(num_classes)]
