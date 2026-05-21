import copy
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import (
    build_cr_datasets,
    count_by_class,
    make_domain_loaders,
    make_loader,
)
from .losses import (
    l1_recon,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
    margin_recon_loss,
    supcon_loss,
)
from .metrics import confusion_matrix_np, per_class_accuracy, save_confusion_outputs
from .models import PatchDiscriminator, STFTDDGModel
from .train_utils import (
    append_csv,
    evaluate_classifier,
    maybe_limit_steps,
    paired_domain_steps,
    save_checkpoint,
    save_json,
    set_seed,
    steps_per_epoch,
)


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def _batch_xy(batch, device):
    if len(batch) == 3:
        x, y, pos = batch
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True), pos.to(
            device, non_blocking=True
        )
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), None


@torch.no_grad()
def evaluate_reconstruction(model: STFTDDGModel, loaders, device, limit_batches: int = 0):
    model.eval()
    total_loss = 0.0
    total = 0
    batches = 0
    for _, loader in loaders:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            recon = model.reconstruct(x)
            loss = torch.abs(recon - x).flatten(1).mean(dim=1)
            total_loss += float(loss.sum().item())
            total += int(x.shape[0])
            batches += 1
            if limit_batches and batches >= limit_batches:
                break
        if limit_batches and batches >= limit_batches:
            break
    return {"recon_l1": total_loss / max(total, 1), "total": total}


@torch.no_grad()
def evaluate_source_domains(model: STFTDDGModel, val_loaders, device, cfg: ExperimentConfig):
    domain_rows = {}
    total_loss = 0.0
    total_count = 0
    acc_values = []
    for rx, loader in val_loaders:
        stats = evaluate_classifier(model, [loader], device, cfg.num_classes, cfg.limit_val_batches)
        domain_rows[rx] = stats
        total_loss += stats["loss"] * stats["total"]
        total_count += stats["total"]
        acc_values.append(stats["acc"])
    mean_acc = sum(acc_values) / max(len(acc_values), 1)
    worst_acc = min(acc_values) if acc_values else 0.0
    return {
        "loss": total_loss / max(total_count, 1),
        "acc": mean_acc,
        "worst_acc": worst_acc,
        "domains": domain_rows,
    }


def _stage1_best_score(val_stats: Dict, cfg: ExperimentConfig) -> float:
    if cfg.stage1_best_metric == "mean":
        return val_stats["acc"]
    if cfg.stage1_best_metric == "worst":
        return val_stats["worst_acc"]
    raise ValueError(f"Unknown stage1_best_metric={cfg.stage1_best_metric!r}")


def _load_identity_pretrain(model: STFTDDGModel, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"]
    current = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if k in current and (k.startswith("eid.") or k.startswith("classifier."))
    }
    current.update(filtered)
    model.load_state_dict(current)


def train_stage0_pretrain(
    target_rx: str, cfg: ExperimentConfig, run_dir: Path, device: torch.device
) -> Path:
    """Supervised source-domain pretraining for Eid+C before generator training."""
    set_seed(cfg.seed - 1)
    cr = build_cr_datasets(target_rx, cfg, return_pos=False)
    train_loaders = make_domain_loaders(cr.source_train, cfg, shuffle=True)
    val_loaders = make_domain_loaders(cr.source_val, cfg, shuffle=False)
    model = STFTDDGModel(
        cfg.num_classes,
        cfg.z_id_dim,
        cfg.z_var_channels,
        arc_margin_m=cfg.arc_margin_m,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.eid.parameters()) + list(model.classifier.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    log_path = run_dir / target_rx / "logs" / "stage0_pretrain_log.csv"
    ckpt_dir = run_dir / target_rx / "checkpoints"
    best_ckpt = ckpt_dir / "stage0_pretrain_best.pt"
    best_val_acc = -1.0
    patience_counter = 0
    steps = maybe_limit_steps(steps_per_epoch(train_loaders), cfg.limit_train_batches)

    for epoch in range(1, cfg.stage0_pretrain_epochs + 1):
        model.train()
        loader_iters = [iter(loader) for _, loader in train_loaders]
        totals = {"loss": 0.0, "loss_scl": 0.0, "correct": 0, "count": 0}
        start_time = time.time()
        for _ in range(steps):
            xs = []
            ys = []
            for idx, (_, loader) in enumerate(train_loaders):
                try:
                    batch = next(loader_iters[idx])
                except StopIteration:
                    loader_iters[idx] = iter(loader)
                    batch = next(loader_iters[idx])
                x, y, _ = _batch_xy(batch, device)
                xs.append(x)
                ys.append(y)
            x_all = torch.cat(xs, dim=0)
            y_all = torch.cat(ys, dim=0)
            optimizer.zero_grad(set_to_none=True)
            z_id = model.eid(x_all)
            logits = model.classifier(z_id, y_all)
            loss_cls = F.cross_entropy(logits, y_all)
            loss_scl = supcon_loss(z_id, y_all, cfg.scl_temp)
            loss = loss_cls + cfg.scl_w * loss_scl
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            totals["loss"] += float(loss.item())
            totals["loss_scl"] += float(loss_scl.item())
            totals["correct"] += int((pred == y_all).sum().item())
            totals["count"] += int(y_all.numel())

        val_stats = evaluate_classifier(model, val_loaders, device, cfg.num_classes, cfg.limit_val_batches)
        is_best = val_stats["acc"] > best_val_acc
        if is_best:
            best_val_acc = val_stats["acc"]
            patience_counter = 0
            save_checkpoint(
                best_ckpt,
                model,
                stage="stage0_pretrain",
                target_rx=target_rx,
                epoch=epoch,
                best_val_acc=best_val_acc,
            )
        else:
            patience_counter += 1

        train_acc = totals["correct"] / max(totals["count"], 1)
        row = {
            "target_rx": target_rx,
            "epoch": epoch,
            "train_loss": totals["loss"] / max(steps, 1),
            "train_loss_scl": totals["loss_scl"] / max(steps, 1),
            "train_acc": train_acc,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - start_time,
            "early_stop_counter": patience_counter,
            "is_best": int(is_best),
        }
        append_csv(log_path, row)
        print(
            f"[Stage0Pretrain][{target_rx}] epoch={epoch} "
            f"loss={row['train_loss']:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_stats['acc']:.4f} best={int(is_best)}",
            flush=True,
        )
        if patience_counter >= cfg.pretrain_patience:
            break

    save_checkpoint(ckpt_dir / "stage0_pretrain_last.pt", model, stage="stage0_pretrain", target_rx=target_rx)
    return best_ckpt


def _stage0_forward(model: STFTDDGModel, x_a: torch.Tensor, x_b: torch.Tensor):
    s_a, f_a = model.encode_pair(x_a)
    s_b, f_b = model.encode_pair(x_b)
    x_ba = model.generator(s_b, f_a)
    x_ab = model.generator(s_a, f_b)
    x_a_recon = model.generator(s_a, f_a)
    x_b_recon = model.generator(s_b, f_b)
    # DDG-style cycle: recover x_a with variation from x_ab and identity from x_ba,
    # and recover x_b with variation from x_ba and identity from x_ab.
    s_a_recon, _ = model.encode_pair(x_ab.detach())
    _, f_a_recon = model.encode_pair(x_ba.detach())
    s_b_recon, _ = model.encode_pair(x_ba.detach())
    _, f_b_recon = model.encode_pair(x_ab.detach())
    x_aba = model.generator(s_a_recon, f_a_recon)
    x_bab = model.generator(s_b_recon, f_b_recon)
    return {
        "s_a": s_a,
        "s_b": s_b,
        "f_a": f_a,
        "f_b": f_b,
        "x_ba": x_ba,
        "x_ab": x_ab,
        "x_a_recon": x_a_recon,
        "x_b_recon": x_b_recon,
        "x_aba": x_aba,
        "x_bab": x_bab,
    }


def train_stage0(
    target_rx: str,
    cfg: ExperimentConfig,
    run_dir: Path,
    device: torch.device,
    identity_ckpt: Path,
) -> Path:
    set_seed(cfg.seed)
    cr = build_cr_datasets(target_rx, cfg, return_pos=False)
    train_loaders = make_domain_loaders(cr.source_train, cfg, shuffle=True)
    val_loaders = make_domain_loaders(cr.source_val, cfg, shuffle=False)
    model = STFTDDGModel(
        cfg.num_classes,
        cfg.z_id_dim,
        cfg.z_var_channels,
        arc_margin_m=cfg.arc_margin_m,
    ).to(device)
    _load_identity_pretrain(model, identity_ckpt, device)
    disc = PatchDiscriminator().to(device)
    _set_requires_grad(model.eid, False)
    _set_requires_grad(model.classifier, False)

    opt_g = torch.optim.Adam(
        list(model.evar.parameters()) + list(model.generator.parameters()),
        lr=cfg.lr,
        betas=(0.0, 0.999),
        weight_decay=cfg.weight_decay,
    )
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.lr_d, weight_decay=cfg.weight_decay)

    log_path = run_dir / target_rx / "logs" / "stage0_train_log.csv"
    ckpt_dir = run_dir / target_rx / "checkpoints"
    save_json(
        run_dir / target_rx / "data_summary_stage0.json",
        {
            "target_rx": target_rx,
            "source_train_class_counts": {
                rx: count_by_class(ds, cfg.num_classes) for rx, ds in cr.source_train
            },
            "source_val_class_counts": {
                rx: count_by_class(ds, cfg.num_classes) for rx, ds in cr.source_val
            },
            "target_test_class_counts": count_by_class(cr.target_test, cfg.num_classes),
            "stft_shape": cfg.stft_shape,
        },
    )

    best_val = float("inf")
    best_ckpt = ckpt_dir / "stage0_best.pt"
    patience_counter = 0
    steps = maybe_limit_steps(steps_per_epoch(train_loaders), cfg.limit_train_batches)

    for epoch in range(1, cfg.stage0_epochs + 1):
        model.train()
        model.eid.eval()
        model.classifier.eval()
        disc.train()
        gen = paired_domain_steps(train_loaders)
        epoch_stats = {
            "loss_total": 0.0,
            "loss_d": 0.0,
            "loss_adv": 0.0,
            "loss_recon_x": 0.0,
            "loss_cyc": 0.0,
        }
        start_time = time.time()
        cyc_w = min(
            cfg.max_cyc_w,
            cfg.recon_x_cyc_w
            + max(0, epoch - int(cfg.stage0_epochs * cfg.warm_iter_r)) * cfg.warm_scale,
        )
        for _ in range(steps):
            _, batch_a, _, batch_b = next(gen)
            x_a, _, _ = _batch_xy(batch_a, device)
            x_b, _, _ = _batch_xy(batch_b, device)

            out = _stage0_forward(model, x_a, x_b)

            _set_requires_grad(disc, True)
            opt_d.zero_grad(set_to_none=True)
            real_preds = disc(torch.cat([x_a, x_b], dim=0))
            fake_preds = disc(torch.cat([out["x_ba"].detach(), out["x_ab"].detach()], dim=0))
            loss_d = lsgan_discriminator_loss(real_preds, fake_preds)
            loss_d.backward()
            opt_d.step()

            _set_requires_grad(disc, False)
            opt_g.zero_grad(set_to_none=True)
            fake_preds_g = disc(torch.cat([out["x_ba"], out["x_ab"]], dim=0))
            loss_adv = lsgan_generator_loss(fake_preds_g)
            loss_recon_x = l1_recon(out["x_a_recon"], x_a) + l1_recon(out["x_b_recon"], x_b)
            loss_cyc = l1_recon(out["x_aba"], x_a) + l1_recon(out["x_bab"], x_b)
            loss_total = cfg.gan_w * loss_adv + cfg.recon_x_w * loss_recon_x + cyc_w * loss_cyc
            loss_total.backward()
            opt_g.step()
            _set_requires_grad(disc, True)

            epoch_stats["loss_total"] += float(loss_total.item())
            epoch_stats["loss_d"] += float(loss_d.item())
            epoch_stats["loss_adv"] += float(loss_adv.item())
            epoch_stats["loss_recon_x"] += float(loss_recon_x.item())
            epoch_stats["loss_cyc"] += float(loss_cyc.item())

        for k in epoch_stats:
            epoch_stats[k] /= max(steps, 1)
        val_stats = evaluate_reconstruction(model, val_loaders, device, cfg.limit_val_batches)
        is_best = val_stats["recon_l1"] < best_val
        if is_best:
            best_val = val_stats["recon_l1"]
            patience_counter = 0
            save_checkpoint(
                best_ckpt,
                model,
                stage="stage0",
                target_rx=target_rx,
                epoch=epoch,
                best_val_recon=best_val,
                identity_ckpt=str(identity_ckpt),
            )
        else:
            patience_counter += 1

        row = {
            "target_rx": target_rx,
            "epoch": epoch,
            **epoch_stats,
            "val_recon_l1": val_stats["recon_l1"],
            "cyc_w": cyc_w,
            "lr_g": opt_g.param_groups[0]["lr"],
            "lr_d": opt_d.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - start_time,
            "early_stop_counter": patience_counter,
            "is_best": int(is_best),
        }
        append_csv(log_path, row)
        print(
            f"[Stage0][{target_rx}] epoch={epoch} "
            f"loss={row['loss_total']:.4f} d={row['loss_d']:.4f} "
            f"recon={row['loss_recon_x']:.4f} cyc={row['loss_cyc']:.4f} "
            f"val_recon={row['val_recon_l1']:.4f} best={int(is_best)}",
            flush=True,
        )
        if patience_counter >= cfg.patience:
            break

    save_checkpoint(
        ckpt_dir / "stage0_last.pt",
        model,
        stage="stage0",
        target_rx=target_rx,
        identity_ckpt=str(identity_ckpt),
    )
    return best_ckpt


def _load_stage0(model: STFTDDGModel, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"]
    current = model.state_dict()
    filtered = {
        k: v
        for k, v in state.items()
        if k in current
        and (
            k.startswith("eid.")
            or k.startswith("evar.")
            or k.startswith("generator.")
            or k.startswith("classifier.")
        )
    }
    current.update(filtered)
    model.load_state_dict(current)


def train_stage1(
    target_rx: str,
    cfg: ExperimentConfig,
    run_dir: Path,
    device: torch.device,
    stage0_ckpt: Path,
) -> Path:
    set_seed(cfg.seed + 1)
    cr = build_cr_datasets(target_rx, cfg, return_pos=True)
    train_loaders = make_domain_loaders(cr.source_train, cfg, shuffle=True)
    val_loaders = make_domain_loaders(cr.source_val, cfg, shuffle=False)

    model = STFTDDGModel(
        cfg.num_classes,
        cfg.z_id_dim,
        cfg.z_var_channels,
        arc_margin_m=cfg.arc_margin_m,
    ).to(device)
    _load_stage0(model, stage0_ckpt, device)
    pretrain_model = copy.deepcopy(model).to(device)
    pretrain_model.eval()
    for p in pretrain_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    log_path = run_dir / target_rx / "logs" / "stage1_train_log.csv"
    ckpt_dir = run_dir / target_rx / "checkpoints"
    best_ckpt = ckpt_dir / "stage1_best.pt"
    best_stage1_score = -1.0
    patience_counter = 0
    steps = maybe_limit_steps(steps_per_epoch(train_loaders), cfg.limit_train_batches)

    for epoch in range(1, cfg.stage1_epochs + 1):
        model.train()
        gen = paired_domain_steps(train_loaders)
        totals = {
            "loss_total": 0.0,
            "loss_cls": 0.0,
            "loss_recon_p": 0.0,
            "loss_recon_id": 0.0,
            "loss_scl": 0.0,
            "train_correct": 0,
            "train_count": 0,
        }
        start_time = time.time()
        for _ in range(steps):
            _, batch_a, _, batch_b = next(gen)
            x_a, y_a, pos_a = _batch_xy(batch_a, device)
            x_b, y_b, pos_b = _batch_xy(batch_b, device)
            assert pos_a is not None and pos_b is not None

            with torch.no_grad():
                s_a_pre, f_a_pre = pretrain_model.encode_pair(x_a)
                s_b_pre, f_b_pre = pretrain_model.encode_pair(x_b)
                x_ba = pretrain_model.generator(s_b_pre, f_a_pre)
                x_ab = pretrain_model.generator(s_a_pre, f_b_pre)

            optimizer.zero_grad(set_to_none=True)
            s_a, z_a = model.encode_pair(x_a)
            s_b, z_b = model.encode_pair(x_b)
            _, z_pa = model.encode_pair(pos_a)
            _, z_pb = model.encode_pair(pos_b)
            x_a_recon_p = model.generator(s_a, z_pa)
            x_b_recon_p = model.generator(s_b, z_pb)

            logits_a = model.classify(x_a, y_a)
            logits_b = model.classify(x_b, y_b)
            logits_pa = model.classify(pos_a, y_a)
            logits_pb = model.classify(pos_b, y_b)
            loss_cls = (
                F.cross_entropy(logits_a, y_a)
                + F.cross_entropy(logits_b, y_b)
                + F.cross_entropy(logits_pa, y_a)
                + F.cross_entropy(logits_pb, y_b)
            )
            loss_recon_p = margin_recon_loss(x_a_recon_p, x_a, cfg.margin) + margin_recon_loss(
                x_b_recon_p, x_b, cfg.margin
            )
            logits_ba = model.classify(x_ba, y_a)
            logits_ab = model.classify(x_ab, y_b)
            loss_recon_id = F.cross_entropy(logits_ba, y_a) + F.cross_entropy(logits_ab, y_b)
            z_all = torch.cat([z_a, z_b, z_pa, z_pb], dim=0)
            y_all = torch.cat([y_a, y_b, y_a, y_b], dim=0)
            loss_scl = supcon_loss(z_all, y_all, cfg.scl_temp)
            loss_total = (
                loss_cls
                + cfg.recon_xp_w * loss_recon_p
                + cfg.recon_id_w * loss_recon_id
                + cfg.scl_w * loss_scl
            )
            loss_total.backward()
            optimizer.step()

            totals["loss_total"] += float(loss_total.item())
            totals["loss_cls"] += float(loss_cls.item())
            totals["loss_recon_p"] += float(loss_recon_p.item())
            totals["loss_recon_id"] += float(loss_recon_id.item())
            totals["loss_scl"] += float(loss_scl.item())
            for logits, y in ((logits_a, y_a), (logits_b, y_b)):
                pred = logits.argmax(dim=1)
                totals["train_correct"] += int((pred == y).sum().item())
                totals["train_count"] += int(y.numel())

        val_stats = evaluate_source_domains(model, val_loaders, device, cfg)
        selected_score = _stage1_best_score(val_stats, cfg)
        is_best = selected_score > best_stage1_score
        if is_best:
            best_stage1_score = selected_score
            patience_counter = 0
            save_checkpoint(
                best_ckpt,
                model,
                stage="stage1",
                target_rx=target_rx,
                epoch=epoch,
                best_val_acc=val_stats["acc"],
                best_val_worst_acc=val_stats["worst_acc"],
                best_stage1_score=best_stage1_score,
                stage1_best_metric=cfg.stage1_best_metric,
                stage0_ckpt=str(stage0_ckpt),
            )
        else:
            patience_counter += 1

        train_acc = totals["train_correct"] / max(totals["train_count"], 1)
        row = {
            "target_rx": target_rx,
            "epoch": epoch,
            "loss_total": totals["loss_total"] / max(steps, 1),
            "loss_cls": totals["loss_cls"] / max(steps, 1),
            "loss_recon_p": totals["loss_recon_p"] / max(steps, 1),
            "loss_recon_id": totals["loss_recon_id"] / max(steps, 1),
            "loss_scl": totals["loss_scl"] / max(steps, 1),
            "train_acc": train_acc,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_worst_acc": val_stats["worst_acc"],
            "stage1_best_metric": cfg.stage1_best_metric,
            "val_selected_score": selected_score,
            **{
                f"val_acc_{rx.replace('-', '_')}": stats["acc"]
                for rx, stats in val_stats["domains"].items()
            },
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - start_time,
            "early_stop_counter": patience_counter,
            "is_best": int(is_best),
        }
        append_csv(log_path, row)
        print(
            f"[Stage1][{target_rx}] epoch={epoch} loss={row['loss_total']:.4f} "
            f"cls={row['loss_cls']:.4f} recon_p={row['loss_recon_p']:.4f} "
            f"recon_id={row['loss_recon_id']:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_stats['acc']:.4f} worst_val={val_stats['worst_acc']:.4f} "
            f"select_{cfg.stage1_best_metric}={selected_score:.4f} best={int(is_best)}",
            flush=True,
        )
        if patience_counter >= cfg.patience:
            break

    save_checkpoint(ckpt_dir / "stage1_last.pt", model, stage="stage1", target_rx=target_rx)
    return best_ckpt


def test_target(
    target_rx: str,
    cfg: ExperimentConfig,
    run_dir: Path,
    device: torch.device,
    stage1_ckpt: Path,
) -> Dict:
    cr = build_cr_datasets(target_rx, cfg, return_pos=False)
    test_loader = make_loader(cr.target_test, cfg, shuffle=False)
    model = STFTDDGModel(
        cfg.num_classes,
        cfg.z_id_dim,
        cfg.z_var_channels,
        arc_margin_m=cfg.arc_margin_m,
    ).to(device)
    ckpt = torch.load(stage1_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    stats = evaluate_classifier(model, [test_loader], device, cfg.num_classes)
    cm = confusion_matrix_np(stats["y_true"], stats["y_pred"], cfg.num_classes)
    out_paths = save_confusion_outputs(cm, run_dir / target_rx / "evaluation", f"STFT_DDG_CR_{target_rx}")
    per_class = per_class_accuracy(cm).tolist()
    result = {
        "target_rx": target_rx,
        "test_acc": stats["acc"],
        "test_loss": stats["loss"],
        "test_total": stats["total"],
        "per_class_acc": per_class,
        "confusion_matrix": cm.tolist(),
        "stage1_ckpt": str(stage1_ckpt),
        "outputs": {k: str(v) for k, v in out_paths.items()},
    }
    save_json(run_dir / target_rx / "evaluation" / "result.json", result)
    print(
        f"[Test][{target_rx}] overall_acc={stats['acc']:.6f} "
        f"per_class={[round(v, 4) for v in per_class]}",
        flush=True,
    )
    return result


def run_target(target_rx: str, cfg: ExperimentConfig, run_dir: Path, device: torch.device) -> Dict:
    target_dir = run_dir / target_rx
    target_dir.mkdir(parents=True, exist_ok=True)
    identity_ckpt = train_stage0_pretrain(target_rx, cfg, run_dir, device)
    stage0_ckpt = train_stage0(target_rx, cfg, run_dir, device, identity_ckpt)
    stage1_ckpt = train_stage1(target_rx, cfg, run_dir, device, stage0_ckpt)
    result = test_target(target_rx, cfg, run_dir, device, stage1_ckpt)
    save_json(target_dir / "result.json", result)
    return result
