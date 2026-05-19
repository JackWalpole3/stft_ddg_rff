import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import accuracy_from_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class CyclingLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def paired_domain_steps(loaders: List[Tuple[str, object]]) -> Iterator[Tuple[str, object, str, object]]:
    cycling = [(rx, CyclingLoader(loader)) for rx, loader in loaders]
    step = 0
    while True:
        i = step % len(cycling)
        j = (step + 1) % len(cycling)
        rx_a, loader_a = cycling[i]
        rx_b, loader_b = cycling[j]
        yield rx_a, loader_a.next(), rx_b, loader_b.next()
        step += 1


def steps_per_epoch(loaders: List[Tuple[str, object]]) -> int:
    return min(len(loader) for _, loader in loaders)


def maybe_limit_steps(total_steps: int, limit: int) -> int:
    if limit and limit > 0:
        return min(total_steps, limit)
    return total_steps


def append_csv(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


@torch.no_grad()
def evaluate_classifier(model, loaders, device, num_classes: int, limit_batches: int = 0):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    seen_batches = 0
    for loader in loaders:
        actual_loader = loader[1] if isinstance(loader, tuple) else loader
        for x, y in actual_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model.classify(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            correct, count = accuracy_from_logits(logits, y)
            total_loss += float(loss.item())
            total_correct += correct
            total += count
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            seen_batches += 1
            if limit_batches and seen_batches >= limit_batches:
                break
        if limit_batches and seen_batches >= limit_batches:
            break
    mean_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return {"loss": mean_loss, "acc": acc, "total": total, "y_true": y_true, "y_pred": y_pred}


def save_checkpoint(path: Path, model, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    payload.update(extra)
    torch.save(payload, path)

