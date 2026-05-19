from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class ExperimentConfig:
    project_name: str = "STFT-DDG-RFF"
    data_path: str = "/home/wangfengsheng/WiSig/ManySig.pkl/ManySig.pkl"
    output_dir: str = "/home/wangfengsheng/STFT-DDG-RFF/runs"
    receivers: Tuple[str, ...] = ("1-1", "1-19", "14-7", "18-2")
    num_classes: int = 6
    samples_per_tx: int = 500
    train_per_tx: int = 315
    val_per_tx: int = 135
    test_per_tx: int = 50
    day_idx: int = 0
    eq_idx: int = 1
    seed: int = 2026

    n_fft: int = 64
    win_length: int = 64
    hop_length: int = 8
    center: bool = False

    batch_size: int = 64
    num_workers: int = 0
    z_id_dim: int = 256
    z_var_channels: int = 128
    lr: float = 1e-4
    lr_d: float = 1e-4
    weight_decay: float = 0.0

    stage0_pretrain_epochs: int = 60
    stage0_epochs: int = 80
    stage1_epochs: int = 120
    patience: int = 30
    pretrain_patience: int = 15

    gan_w: float = 1.0
    recon_x_w: float = 0.5
    recon_x_cyc_w: float = 0.0
    max_cyc_w: float = 2.0
    warm_iter_r: float = 0.2
    warm_scale: float = 5e-3

    margin: float = 0.25
    recon_xp_w: float = 0.1
    recon_id_w: float = 0.1

    limit_train_batches: int = 0
    limit_val_batches: int = 0

    def run_root(self, run_name: str) -> Path:
        return Path(self.output_dir) / run_name

    @property
    def stft_shape(self) -> Tuple[int, int, int]:
        frames = (256 - self.n_fft) // self.hop_length + 1
        return 2, self.n_fft, frames
