from typing import Iterable, List

import torch
import torch.nn.functional as F


def _as_list(preds):
    return preds if isinstance(preds, (list, tuple)) else [preds]


def lsgan_discriminator_loss(real_preds, fake_preds) -> torch.Tensor:
    loss = 0.0
    for pred in _as_list(real_preds):
        loss = loss + F.mse_loss(pred, torch.ones_like(pred))
    for pred in _as_list(fake_preds):
        loss = loss + F.mse_loss(pred, torch.zeros_like(pred))
    return loss / (len(_as_list(real_preds)) + len(_as_list(fake_preds)))


def lsgan_generator_loss(fake_preds) -> torch.Tensor:
    preds = _as_list(fake_preds)
    loss = 0.0
    for pred in preds:
        loss = loss + F.mse_loss(pred, torch.ones_like(pred))
    return loss / len(preds)


def l1_recon(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(input - target.detach()))


def l1_recon_per_sample(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(input - target.detach())
    return diff.flatten(1).mean(dim=1)


def margin_recon_loss(input: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    per_sample = l1_recon_per_sample(input, target)
    return torch.relu(per_sample - margin).mean()


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if z.ndim != 2:
        z = z.flatten(1)
    z = F.normalize(z, dim=1)
    y = y.view(-1, 1)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values
    mask = torch.eq(y, y.T).float()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_count = mask.sum(dim=1)
    mean_log_prob = (mask * log_prob).sum(dim=1) / torch.clamp(pos_count, min=1.0)
    valid = pos_count > 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=z.device)
    return (-mean_log_prob[valid]).mean()

