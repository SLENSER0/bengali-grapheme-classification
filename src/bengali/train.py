"""Fine-tune a CNN with three heads on one fold.

Usage:
    python -m bengali.train --config configs/baseline.yaml --fold 0 --seed 42
    python -m bengali.train --config configs/baseline.yaml --fold 0 --set train.epochs=1

Protocol guarantees: fixed number of epochs (no early stopping / checkpoint
picking on validation), validation is only *logged* per epoch for learning
curves; the reported numbers come from the final epoch.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from .metric import slice_scores
from .models import DinoV3PartialFT, FinetuneNet, count_params
from .runs import load_fold_data, prediction_frame, run_dir, save_run
from .utils import COMPONENTS, Timer, get_device, load_config, resolve, save_json, seed_everything


def autocast_ctx(device: torch.device, amp: str | None):
    if not amp or amp == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def predict(model, imgs: np.ndarray, idx: np.ndarray, device, batch_size: int, amp) -> tuple[dict, dict]:
    model.eval()
    preds = {c: [] for c in COMPONENTS}
    confs = {c: [] for c in COMPONENTS}
    for s in range(0, len(idx), batch_size):
        x = torch.from_numpy(imgs[idx[s : s + batch_size]]).to(device)
        with autocast_ctx(device, amp):
            out = model(x)
        for c in COMPONENTS:
            p = out[c].float().softmax(1)
            conf, arg = p.max(1)
            preds[c].append(arg.cpu().numpy())
            confs[c].append(conf.cpu().numpy())
    model.train()
    return ({c: np.concatenate(v) for c, v in preds.items()}, {c: np.concatenate(v) for c, v in confs.items()})


def run(cfg: dict, fold: int, seed: int) -> dict:
    seed_everything(seed)
    device = get_device(cfg.get("device", "auto"))
    tcfg = cfg["train"]
    imgs, folds, labels, ix = load_fold_data(cfg, fold)
    tr = ix["train"]
    print(f"[fold {fold} seed {seed}] device={device} train={len(tr)} val_id={len(ix['val_id'])} val_ood={len(ix['val_ood'])}")

    mcfg = cfg["model"]
    if mcfg.get("kind", "timm_cnn") == "dinov3_partial":
        model = DinoV3PartialFT(mcfg["backbone"], mcfg["unfreeze_last"], mcfg["input_size"], mcfg.get("invert", True), mcfg.get("dropout", 0.0)).to(device)
        # heads at the base lr, unfrozen ViT blocks at a smaller lr (standard for fine-tuning pretrained transformers)
        groups = [
            {"params": list(model.head.parameters()), "lr": tcfg["lr"]},
            {"params": model.backbone_parameters(), "lr": tcfg["lr"] * tcfg.get("backbone_lr_mult", 1.0)},
        ]
    else:
        model = FinetuneNet(mcfg["backbone"], mcfg.get("pretrained", True), mcfg.get("dropout", 0.0)).to(device)
        groups = [{"params": list(model.parameters()), "lr": tcfg["lr"]}]
    opt = torch.optim.AdamW(groups, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    bs = tcfg["batch_size"]
    # A tiny trailing batch (e.g. 1 image when len(tr) % bs == 1) gives noisy BatchNorm running statistics right
    # before evaluation, so a remainder smaller than `min_last_batch` is skipped (different images each epoch).
    n_full, rem = divmod(len(tr), bs)
    keep_last = rem >= max(2, tcfg.get("min_last_batch", bs // 2))
    steps_per_epoch = n_full + int(keep_last)
    total_steps = tcfg["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=total_steps, pct_start=tcfg.get("warmup_frac", 0.1), anneal_strategy="cos"
    )
    w = tcfg.get("loss_weights", [1.0, 1.0, 1.0])
    amp = tcfg.get("amp", "none") if device.type != "cpu" else "none"
    rng = np.random.default_rng(seed)
    y_all = torch.from_numpy(labels)
    history = []
    timer = Timer()
    for epoch in range(tcfg["epochs"]):
        perm = rng.permutation(tr)
        run_loss, n_seen = 0.0, 0
        t_ep = Timer()
        for s in range(0, steps_per_epoch * bs, bs):
            bidx = np.sort(perm[s : s + bs])  # sorted gather is faster; order inside a batch is irrelevant
            x = torch.from_numpy(imgs[bidx]).to(device, non_blocking=True)
            y = y_all[bidx].to(device)
            with autocast_ctx(device, amp):
                out = model(x)
                loss = sum(wi * F.cross_entropy(out[c].float(), y[:, j]) for j, (c, wi) in enumerate(zip(COMPONENTS, w)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(bidx)
            n_seen += len(bidx)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / n_seen, "epoch_sec": t_ep(), "lr": sched.get_last_lr()[0]}
        if tcfg.get("log_val_every_epoch", True) or epoch == tcfg["epochs"] - 1:
            preds, confs = {}, {}
            for part in ("val_id", "val_ood"):
                preds[part], confs[part] = predict(model, imgs, ix[part], device, cfg["eval"]["batch_size"], amp)
            pdf = prediction_frame(folds, ix, preds, confs)
            sc = slice_scores(pdf)
            rec.update({k: sc[k] for k in ("id_score", "ood_score", "main_score") if k in sc})
        history.append(rec)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)

    d = run_dir(cfg, fold, seed)
    d.mkdir(parents=True, exist_ok=True)
    save_json(history, d / "history.json")
    if cfg.get("save_checkpoint", True):
        ck = resolve("checkpoints") / cfg["experiment"]
        ck.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ck / f"fold{fold}_seed{seed}.pt")
    extra = {
        "train_sec": timer(),
        "n_params": count_params(model),
        "n_params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "device": str(device),
        "epochs": tcfg["epochs"],
        "n_train": int(len(tr)),
    }
    m = save_run(cfg, fold, seed, pdf, extra)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if "score" in k})
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, nargs="+", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.epochs=1")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    for seed in a.seed or cfg.get("seeds", [42]):
        for fold in a.fold if a.fold is not None else cfg.get("folds", [0]):
            run(cfg, fold, seed)


if __name__ == "__main__":
    main()
